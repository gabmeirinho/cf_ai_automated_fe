import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage
} from "ai";
import { createWorkersAI } from "workers-ai-provider";
import { z } from "zod";

const DEFAULT_AI_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8";

function inlineDataUrls(messages: ModelMessage[]): ModelMessage[] {
  return messages.map((message) => {
    if (message.role !== "user" || typeof message.content === "string") {
      return message;
    }

    return {
      ...message,
      content: message.content.map((part) => {
        if (part.type !== "file" || typeof part.data !== "string") return part;

        const match = part.data.match(/^data:([^;]+);base64,(.+)$/);
        if (!match) return part;

        return {
          ...part,
          data: Uint8Array.from(atob(match[2]), (char) => char.charCodeAt(0)),
          mediaType: match[1]
        };
      })
    };
  });
}

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }

        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const workersai = createWorkersAI({ binding: this.env.AI });
    const mcpTools = this.mcp.getAITools();
    const model = this.env.AI_MODEL || DEFAULT_AI_MODEL;
    const messages = inlineDataUrls(
      await convertToModelMessages(this.messages)
    );

    const result = streamText({
      model: workersai(model, {
        sessionAffinity: this.sessionAffinity
      }),
      system: `You are a data-preparation assistant embedded in Automated FE. Help users inspect CSVs, reason about preprocessing choices, and plan frontend automation work. Keep answers concise and actionable.

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task or reminder, use the schedule tool.`,
      messages: pruneMessages({
        messages,
        toolCalls: "before-last-2-messages"
      }),
      tools: {
        ...mcpTools,
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser when local time matters.",
          inputSchema: z.object({})
        }),
        calculate: tool({
          description: "Perform a math calculation with two numbers.",
          inputSchema: z.object({
            a: z.number(),
            b: z.number(),
            operator: z.enum(["+", "-", "*", "/", "%"])
          }),
          execute: async ({ a, b, operator }) => {
            if (operator === "/" && b === 0)
              return { error: "Division by zero" };

            const operations = {
              "+": a + b,
              "-": a - b,
              "*": a * b,
              "/": a / b,
              "%": a % b
            };

            return {
              expression: `${a} ${operator} ${b}`,
              result: operations[operator]
            };
          }
        }),
        scheduleTask: tool({
          description:
            "Schedule a task to be executed later. Use this for reminders.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule")
              return "Not a valid schedule input";

            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;

            if (!input) return "Invalid schedule type";

            this.schedule(input, "executeTask", description, {
              idempotent: true
            });

            return `Task scheduled: "${description}" (${when.type}: ${input})`;
          }
        }),
        getScheduledTasks: tool({
          description: "List all scheduled tasks.",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),
        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by ID.",
          inputSchema: z.object({
            taskId: z.string()
          }),
          execute: async ({ taskId }) => {
            this.cancelSchedule(taskId);
            return `Task ${taskId} cancelled.`;
          }
        })
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
