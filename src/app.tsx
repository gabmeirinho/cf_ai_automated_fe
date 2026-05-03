import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart, type UIMessage } from "ai";
import type { ChatAgent } from "./server";
import {
  MAX_CSV_SIZE_BYTES,
  applyPreprocessingStepsToPreviewRows,
  buildAiReviewProfile,
  buildDatasetIntent,
  describePreprocessingStep,
  formatBytes,
  parseCsvFile,
  renderPreviewValue,
  validateCsvFile,
  type ColumnSummary,
  type DatasetSummary,
  type ColumnAssumption,
  type LlmPreprocessingPlan,
  type DatasetIntent,
  type IntentOverride,
  type ProfilingNoteCode,
  type PreprocessingStep,
  type PreprocessingStatus,
  type TransformationDecision
} from "./csv-profile";
import {
  Badge,
  Button,
  Empty,
  InputArea,
  Surface,
  Text
} from "@cloudflare/kumo";
import { Toasty, useKumoToastManager } from "@cloudflare/kumo/components/toast";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import {
  BrainIcon,
  CaretDownIcon,
  CaretUpIcon,
  ChatCircleDotsIcon,
  CheckCircleIcon,
  GearIcon,
  MagnifyingGlassIcon,
  MoonIcon,
  PaperclipIcon,
  PaperPlaneRightIcon,
  SunIcon,
  StopIcon,
  XCircleIcon,
  XIcon,
  TrashIcon
} from "@phosphor-icons/react";

type UploadState =
  | { status: "idle" }
  | { status: "validating"; fileName: string }
  | { status: "parsing"; fileName: string }
  | { status: "ready"; summary: DatasetSummary }
  | { status: "error"; message: string };

const PREVIEW_VISIBLE_COLUMN_COUNT = 10;

type ColumnFilter =
  | "all"
  | "has_notes"
  | "high_missingness"
  | "empty"
  | "constant"
  | "likely_identifier";

type ColumnSortKey =
  | "name"
  | "type"
  | "missingCount"
  | "missingPercent"
  | "nonMissingCount"
  | "uniqueCount"
  | "uniqueRatio"
  | "notesCount";

type SortDirection = "asc" | "desc";

interface ColumnSort {
  key: ColumnSortKey;
  direction: SortDirection;
}

type AiReviewState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; plan: LlmPreprocessingPlan }
  | { status: "error"; message: string };

const COLUMN_FILTERS: Array<{
  key: ColumnFilter;
  label: string;
}> = [
  { key: "all", label: "All" },
  { key: "has_notes", label: "Has notes" },
  { key: "high_missingness", label: "High missingness" },
  { key: "empty", label: "Empty" },
  { key: "constant", label: "Constant" },
  { key: "likely_identifier", label: "Likely ID" }
];

function formatPercent(value: number) {
  return `${value.toFixed(value < 10 && value > 0 ? 1 : 0)}%`;
}

function hasProfilingNote(column: ColumnSummary, code: ProfilingNoteCode) {
  return column.profilingNotes.some((note) => note.code === code);
}

function getMissingPercent(column: ColumnSummary, rowCount: number) {
  return rowCount === 0 ? 0 : (column.missingCount / rowCount) * 100;
}

function getColumnSortValue(
  column: ColumnSummary,
  key: ColumnSortKey,
  rowCount: number
) {
  switch (key) {
    case "name":
      return column.name.toLowerCase();
    case "type":
      return column.inferredType;
    case "missingCount":
      return column.missingCount;
    case "missingPercent":
      return getMissingPercent(column, rowCount);
    case "nonMissingCount":
      return column.nonMissingCount;
    case "uniqueCount":
      return column.uniqueCount;
    case "uniqueRatio":
      return column.uniqueRatio;
    case "notesCount":
      return column.profilingNotes.length;
  }
}

function filterColumn(column: ColumnSummary, filter: ColumnFilter) {
  switch (filter) {
    case "all":
      return true;
    case "has_notes":
      return column.profilingNotes.length > 0;
    case "high_missingness":
      return hasProfilingNote(column, "high_missingness");
    case "empty":
      return hasProfilingNote(column, "empty_column");
    case "constant":
      return hasProfilingNote(column, "constant_column");
    case "likely_identifier":
      return hasProfilingNote(column, "likely_identifier");
  }
}

function getEffectivePreprocessingStep(
  step: PreprocessingStep,
  statuses: Record<string, PreprocessingStatus>
) {
  return {
    ...step,
    status: statuses[step.id] ?? step.status
  } satisfies PreprocessingStep;
}

function ThemeToggle() {
  const [dark, setDark] = useState(
    () => document.documentElement.getAttribute("data-mode") === "dark"
  );

  const toggle = useCallback(() => {
    const next = !dark;
    setDark(next);
    const mode = next ? "dark" : "light";
    document.documentElement.setAttribute("data-mode", mode);
    document.documentElement.style.colorScheme = mode;
    localStorage.setItem("theme", mode);
  }, [dark]);

  return (
    <Button
      variant="secondary"
      shape="square"
      icon={dark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
      onClick={toggle}
      aria-label="Toggle theme"
    />
  );
}

interface Attachment {
  id: string;
  file: File;
  preview: string;
  mediaType: string;
}

function createAttachment(file: File): Attachment {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    file,
    preview: URL.createObjectURL(file),
    mediaType: file.type || "application/octet-stream"
  };
}

function fileToDataUri(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function ToolPartView({
  part,
  addToolApprovalResponse
}: {
  part: UIMessage["parts"][number];
  addToolApprovalResponse: (response: {
    id: string;
    approved: boolean;
  }) => void;
}) {
  if (!isToolUIPart(part)) return null;

  const toolName = getToolName(part);

  if (part.state === "output-available") {
    return (
      <div className="rounded-md border border-kumo-line bg-kumo-elevated px-3 py-2">
        <div className="mb-1 flex items-center gap-2">
          <GearIcon size={14} className="text-kumo-inactive" />
          <Text size="xs" variant="secondary" bold>
            {toolName}
          </Text>
          <Badge variant="secondary">Done</Badge>
        </div>
        <pre className="max-h-32 overflow-auto whitespace-pre-wrap text-xs text-kumo-subtle">
          {JSON.stringify(part.output, null, 2)}
        </pre>
      </div>
    );
  }

  if ("approval" in part && part.state === "approval-requested") {
    const approvalId = (part.approval as { id?: string })?.id;

    return (
      <div className="rounded-md border border-kumo-warning/50 bg-kumo-warning/10 px-3 py-2">
        <div className="mb-2 flex items-center gap-2">
          <GearIcon size={14} className="text-kumo-warning" />
          <Text size="xs" bold>
            Approval needed: {toolName}
          </Text>
        </div>
        <pre className="mb-2 max-h-32 overflow-auto whitespace-pre-wrap text-xs text-kumo-subtle">
          {JSON.stringify(part.input, null, 2)}
        </pre>
        <div className="flex gap-2">
          <Button
            variant="primary"
            size="sm"
            icon={<CheckCircleIcon size={14} />}
            onClick={() => {
              if (approvalId) {
                addToolApprovalResponse({ id: approvalId, approved: true });
              }
            }}
          >
            Approve
          </Button>
          <Button
            variant="secondary"
            size="sm"
            icon={<XCircleIcon size={14} />}
            onClick={() => {
              if (approvalId) {
                addToolApprovalResponse({ id: approvalId, approved: false });
              }
            }}
          >
            Reject
          </Button>
        </div>
      </div>
    );
  }

  if (
    part.state === "output-denied" ||
    ("approval" in part &&
      (part.approval as { approved?: boolean })?.approved === false)
  ) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-kumo-line bg-kumo-elevated px-3 py-2">
        <XCircleIcon size={14} className="text-kumo-danger" />
        <Text size="xs" variant="secondary" bold>
          {toolName}
        </Text>
        <Badge variant="secondary">Rejected</Badge>
      </div>
    );
  }

  if (part.state === "input-available" || part.state === "input-streaming") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-kumo-line bg-kumo-elevated px-3 py-2">
        <GearIcon size={14} className="animate-spin text-kumo-inactive" />
        <Text size="xs" variant="secondary">
          Running {toolName}...
        </Text>
      </div>
    );
  }

  return null;
}

function AgentChatSidebar() {
  const toasts = useKumoToastManager();
  const [connected, setConnected] = useState(false);
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const attachmentPreviewsRef = useRef<string[]>([]);

  const agent = useAgent<ChatAgent>({
    agent: "ChatAgent",
    onOpen: useCallback(() => setConnected(true), []),
    onClose: useCallback(() => setConnected(false), []),
    onError: useCallback(
      (error: Event) => console.error("Agent WebSocket error:", error),
      []
    ),
    onMessage: useCallback(
      (message: MessageEvent) => {
        try {
          const data = JSON.parse(String(message.data));
          if (data.type === "scheduled-task") {
            toasts.add({
              title: "Scheduled task completed",
              description: data.description,
              timeout: 0
            });
          }
        } catch {
          // Ignore non-JSON agent protocol messages.
        }
      },
      [toasts]
    )
  });

  const {
    messages,
    sendMessage,
    clearHistory,
    addToolApprovalResponse,
    stop,
    status
  } = useAgentChat({
    agent,
    onToolCall: async (event) => {
      if (
        "addToolOutput" in event &&
        event.toolCall.toolName === "getUserTimezone"
      ) {
        event.addToolOutput({
          toolCallId: event.toolCall.toolCallId,
          output: {
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            localTime: new Date().toLocaleTimeString()
          }
        });
      }
    }
  });

  const isStreaming = status === "streaming" || status === "submitted";

  useEffect(() => {
    function handleExternalMessage(event: Event) {
      const detail = (event as CustomEvent<{ text?: string }>).detail;
      if (!detail?.text) return;

      sendMessage({
        role: "user",
        parts: [{ type: "text", text: detail.text }]
      });
    }

    window.addEventListener(
      "automated-fe:agent-chat-message",
      handleExternalMessage
    );
    return () =>
      window.removeEventListener(
        "automated-fe:agent-chat-message",
        handleExternalMessage
      );
  }, [sendMessage]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isStreaming) textareaRef.current?.focus();
  }, [isStreaming]);

  useEffect(() => {
    attachmentPreviewsRef.current = attachments.map(
      (attachment) => attachment.preview
    );
  }, [attachments]);

  useEffect(() => {
    return () => {
      attachmentPreviewsRef.current.forEach((preview) =>
        URL.revokeObjectURL(preview)
      );
    };
  }, []);

  const addFiles = useCallback((files: FileList | File[]) => {
    const images = Array.from(files).filter((file) =>
      file.type.startsWith("image/")
    );
    if (images.length === 0) return;
    setAttachments((current) => [...current, ...images.map(createAttachment)]);
  }, []);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((current) => {
      const attachment = current.find((item) => item.id === id);
      if (attachment) URL.revokeObjectURL(attachment.preview);
      return current.filter((item) => item.id !== id);
    });
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if ((!text && attachments.length === 0) || isStreaming) return;

    setInput("");
    const parts: Array<
      | { type: "text"; text: string }
      | { type: "file"; mediaType: string; url: string }
    > = [];

    if (text) parts.push({ type: "text", text });

    for (const attachment of attachments) {
      parts.push({
        type: "file",
        mediaType: attachment.mediaType,
        url: await fileToDataUri(attachment.file)
      });
      URL.revokeObjectURL(attachment.preview);
    }

    setAttachments([]);
    sendMessage({ role: "user", parts });
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [attachments, input, isStreaming, sendMessage]);

  const handlePaste = useCallback(
    (event: React.ClipboardEvent) => {
      const files = Array.from(event.clipboardData.items)
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile())
        .filter((file): file is File => Boolean(file));

      if (files.length > 0) {
        event.preventDefault();
        addFiles(files);
      }
    },
    [addFiles]
  );

  const promptSuggestions = [
    "Summarize preprocessing risks in this CSV.",
    "Which columns should I inspect first?",
    "What timezone am I in?",
    "Remind me in 10 minutes to review the pipeline."
  ];

  return (
    <Surface
      className={`flex h-[calc(100vh-7.5rem)] min-h-[560px] flex-col overflow-hidden rounded-xl ring transition-colors lg:sticky lg:top-5 ${
        isDragging ? "bg-kumo-brand/5 ring-2 ring-kumo-brand" : "ring-kumo-line"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer.types.includes("Files")) setIsDragging(true);
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.currentTarget === event.target) setIsDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        event.stopPropagation();
        setIsDragging(false);
        if (event.dataTransfer.files.length > 0)
          addFiles(event.dataTransfer.files);
      }}
    >
      <div className="border-b border-kumo-line px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ChatCircleDotsIcon size={18} />
              <Text size="sm" bold>
                Agent chat
              </Text>
              <span
                className={`h-2 w-2 rounded-full ${
                  connected ? "bg-kumo-success" : "bg-kumo-inactive"
                }`}
                aria-label={connected ? "Connected" : "Disconnected"}
              />
            </div>
            <Text size="xs" variant="secondary">
              Ask about datasets, preprocessing, and follow-up tasks.
            </Text>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            shape="square"
            icon={<TrashIcon size={14} />}
            aria-label="Clear chat history"
            disabled={messages.length === 0}
            onClick={clearHistory}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <div className="grid gap-3">
            <Empty
              icon={<ChatCircleDotsIcon size={28} />}
              title="No conversation yet"
              contents="Use the agent as a compact workspace companion."
            />
            <div className="grid gap-2">
              {promptSuggestions.map((prompt) => (
                <Button
                  key={prompt}
                  type="button"
                  variant="secondary"
                  size="sm"
                  disabled={!connected || isStreaming}
                  onClick={() =>
                    sendMessage({
                      role: "user",
                      parts: [{ type: "text", text: prompt }]
                    })
                  }
                >
                  {prompt}
                </Button>
              ))}
            </div>
          </div>
        ) : (
          <div className="grid gap-4">
            {messages.map((message: UIMessage, index) => {
              const isUser = message.role === "user";
              const isLastAssistant =
                message.role === "assistant" && index === messages.length - 1;

              return (
                <div key={message.id} className="grid gap-2">
                  {message.parts.filter(isToolUIPart).map((part) => (
                    <ToolPartView
                      key={part.toolCallId}
                      part={part}
                      addToolApprovalResponse={addToolApprovalResponse}
                    />
                  ))}

                  {message.parts
                    .filter(
                      (part) =>
                        part.type === "reasoning" &&
                        (part as { text?: string }).text?.trim()
                    )
                    .map((part, partIndex) => {
                      const reasoning = part as {
                        type: "reasoning";
                        text: string;
                        state?: "streaming" | "done";
                      };
                      const isDone = reasoning.state === "done" || !isStreaming;

                      return (
                        <details
                          key={partIndex}
                          className="rounded-md border border-kumo-line bg-kumo-elevated"
                          open={!isDone}
                        >
                          <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-xs text-kumo-subtle">
                            <BrainIcon size={14} />
                            Reasoning
                            <Badge variant="secondary">
                              {isDone ? "Done" : "Thinking"}
                            </Badge>
                          </summary>
                          <pre className="max-h-40 overflow-auto whitespace-pre-wrap px-3 pb-3 text-xs text-kumo-subtle">
                            {reasoning.text}
                          </pre>
                        </details>
                      );
                    })}

                  {message.parts
                    .filter(
                      (part): part is Extract<typeof part, { type: "file" }> =>
                        part.type === "file" &&
                        (part as { mediaType?: string }).mediaType?.startsWith(
                          "image/"
                        ) === true
                    )
                    .map((part, partIndex) => (
                      <div
                        key={`file-${partIndex}`}
                        className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                      >
                        <img
                          src={part.url}
                          alt="Attachment"
                          className="max-h-48 max-w-[85%] rounded-lg border border-kumo-line object-contain"
                        />
                      </div>
                    ))}

                  {message.parts
                    .filter((part) => part.type === "text")
                    .map((part, partIndex) => {
                      const text = (part as { type: "text"; text: string })
                        .text;
                      if (!text) return null;

                      return (
                        <div
                          key={partIndex}
                          className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                        >
                          {isUser ? (
                            <div className="max-w-[88%] rounded-lg rounded-br-sm bg-kumo-contrast px-3 py-2 text-sm leading-relaxed text-kumo-inverse">
                              {text}
                            </div>
                          ) : (
                            <div className="max-w-[92%] rounded-lg rounded-bl-sm border border-kumo-line bg-kumo-elevated text-sm leading-relaxed text-kumo-default">
                              <Streamdown
                                className="sd-theme rounded-lg p-3"
                                plugins={{ code }}
                                controls={false}
                                isAnimating={isLastAssistant && isStreaming}
                              >
                                {text}
                              </Streamdown>
                            </div>
                          )}
                        </div>
                      );
                    })}
                </div>
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <form
        className="border-t border-kumo-line p-3"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*"
          className="hidden"
          onChange={(event) => {
            if (event.target.files) addFiles(event.target.files);
            event.target.value = "";
          }}
        />

        {attachments.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {attachments.map((attachment) => (
              <div
                key={attachment.id}
                className="group relative overflow-hidden rounded-md border border-kumo-line bg-kumo-elevated"
              >
                <img
                  src={attachment.preview}
                  alt={attachment.file.name}
                  className="h-14 w-14 object-cover"
                />
                <button
                  type="button"
                  className="absolute right-1 top-1 rounded-full bg-kumo-contrast/80 p-0.5 text-kumo-inverse opacity-0 transition-opacity group-hover:opacity-100"
                  aria-label={`Remove ${attachment.file.name}`}
                  onClick={() => removeAttachment(attachment.id)}
                >
                  <XIcon size={10} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2 rounded-lg border border-kumo-line bg-kumo-base p-2 focus-within:border-transparent focus-within:ring-2 focus-within:ring-kumo-ring">
          <Button
            type="button"
            variant="ghost"
            shape="square"
            aria-label="Attach images"
            icon={<PaperclipIcon size={16} />}
            disabled={!connected || isStreaming}
            onClick={() => fileInputRef.current?.click()}
          />
          <InputArea
            ref={textareaRef}
            value={input}
            onValueChange={setInput}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
            onInput={(event) => {
              const element = event.currentTarget;
              element.style.height = "auto";
              element.style.height = `${element.scrollHeight}px`;
            }}
            onPaste={handlePaste}
            placeholder={
              connected ? "Message the agent..." : "Connecting to agent..."
            }
            disabled={!connected || isStreaming}
            rows={1}
            className="max-h-32 flex-1 resize-none bg-transparent! shadow-none! outline-none! ring-0! focus:ring-0!"
          />
          {isStreaming ? (
            <Button
              type="button"
              variant="secondary"
              shape="square"
              aria-label="Stop generation"
              icon={<StopIcon size={16} />}
              onClick={stop}
            />
          ) : (
            <Button
              type="submit"
              variant="primary"
              shape="square"
              aria-label="Send message"
              disabled={
                (!input.trim() && attachments.length === 0) || !connected
              }
              icon={<PaperPlaneRightIcon size={16} />}
            />
          )}
        </div>
      </form>
    </Surface>
  );
}

function PreviewTable({
  columns,
  rows
}: {
  columns: { name: string }[];
  rows: Record<string, string | number | boolean | null>[];
}) {
  const previewWidthPercent =
    (Math.max(columns.length, PREVIEW_VISIBLE_COLUMN_COUNT) /
      PREVIEW_VISIBLE_COLUMN_COUNT) *
    100;

  return (
    <div className="w-full max-w-full overflow-x-auto border-t border-kumo-line">
      <table
        className="table-fixed text-left text-xs"
        style={{ width: `${previewWidthPercent}%` }}
      >
        <thead className="bg-kumo-base text-kumo-subtle">
          <tr>
            {columns.map((column) => (
              <th
                key={column.name}
                className="truncate px-3 py-2 font-medium"
                title={column.name}
              >
                {column.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-kumo-line">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td
                  key={column.name}
                  className="truncate px-3 py-2 text-kumo-subtle"
                  title={renderPreviewValue(row[column.name])}
                >
                  {renderPreviewValue(row[column.name])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function buildReviewChatMessage(
  summary: DatasetSummary,
  plan: LlmPreprocessingPlan
) {
  const acceptedAssumptions = plan.assumptions.filter(
    (assumption) => assumption.status === "accepted"
  );
  const proposedAssumptions = plan.assumptions.filter(
    (assumption) => assumption.status === "proposed"
  );

  return `Use this compact dataset profile and current AI review assumptions as context for our conversation.

Dataset profile:
${JSON.stringify(buildAiReviewProfile(summary), null, 2)}

Current AI review:
${JSON.stringify(
  {
    datasetSummary: plan.datasetSummary,
    acceptedAssumptions,
    proposedAssumptions,
    decisions: plan.decisions,
    globalWarnings: plan.globalWarnings,
    nextQuestions: plan.nextQuestions
  },
  null,
  2
)}

Canonical dataset intent:
${JSON.stringify(buildDatasetIntent(summary, plan.assumptions), null, 2)}

When suggesting changes to assumptions, explain briefly and include an optional JSON patch in this shape:
{
  "updates": [
    {
      "assumptionId": "existing assumption id",
      "status": "accepted | rejected | proposed",
      "role": "feature | target | identifier | timestamp | free_text | ignore | unknown",
      "reason": "short reason"
    }
  ]
}
Do not assume the patch has been applied until I confirm it.`;
}

function updateAssumptionStatus(
  plan: LlmPreprocessingPlan,
  assumptionId: string,
  status: PreprocessingStatus
) {
  return {
    ...plan,
    assumptions: plan.assumptions.map((assumption) =>
      assumption.id === assumptionId ? { ...assumption, status } : assumption
    )
  } satisfies LlmPreprocessingPlan;
}

function buildIntentOverride(
  assumption: ColumnAssumption,
  status: PreprocessingStatus
) {
  return {
    assumptionId: assumption.id,
    columnName: assumption.columnName,
    role: assumption.role,
    status,
    source: "user",
    reason: `User marked ${assumption.columnName} as ${status}.`,
    updatedAt: new Date().toISOString()
  } satisfies IntentOverride;
}

function assumptionRoleLabel(role: ColumnAssumption["role"]) {
  switch (role) {
    case "free_text":
      return "Free text";
    case "identifier":
      return "Identifiers";
    case "timestamp":
      return "Timestamps";
    case "target":
      return "Possible target";
    case "ignore":
      return "Ignore";
    case "feature":
      return "Features";
    case "unknown":
      return "Unknown";
  }
}

function decisionTypeLabel(type: TransformationDecision["type"]) {
  switch (type) {
    case "assumption":
      return "Assumption";
    case "mapping":
      return "Mapping";
    case "normalization":
      return "Normalization";
    case "validation":
      return "Validation";
    case "exclusion":
      return "Exclusion";
  }
}

function DecisionTrace({ decisions }: { decisions: TransformationDecision[] }) {
  if (decisions.length === 0) return null;

  return (
    <div className="rounded-lg border border-kumo-line bg-kumo-base p-3">
      <div className="mb-3">
        <Text size="xs" variant="secondary" bold>
          Decision trace
        </Text>
        <Text size="xs" variant="secondary">
          Inspect the model choices without exposing private reasoning.
        </Text>
      </div>
      <div className="grid gap-2">
        {decisions.map((decision) => (
          <details
            key={decision.id}
            className="rounded-lg border border-kumo-line bg-kumo-elevated"
          >
            <summary className="flex cursor-pointer flex-wrap items-center gap-2 px-3 py-2">
              <Badge variant="secondary">
                {decisionTypeLabel(decision.type)}
              </Badge>
              <Text size="sm" bold>
                {decision.target}
              </Text>
              <Badge
                variant={
                  decision.confidence === "high" ? "primary" : "secondary"
                }
              >
                {decision.confidence}
              </Badge>
            </summary>
            <div className="grid gap-3 border-t border-kumo-line px-3 py-3">
              <div>
                <Text size="xs" variant="secondary" bold>
                  Choice
                </Text>
                <Text size="sm">{decision.decision}</Text>
              </div>
              <div>
                <Text size="xs" variant="secondary" bold>
                  Reason
                </Text>
                <Text size="sm">{decision.reason}</Text>
              </div>
              {(decision.evidence.length > 0 ||
                decision.alternatives.length > 0) && (
                <div className="grid gap-3 sm:grid-cols-2">
                  {decision.evidence.length > 0 && (
                    <div>
                      <Text size="xs" variant="secondary" bold>
                        Evidence
                      </Text>
                      <ul className="mt-1 grid gap-1">
                        {decision.evidence.map((item) => (
                          <li key={item} className="text-xs text-kumo-subtle">
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {decision.alternatives.length > 0 && (
                    <div>
                      <Text size="xs" variant="secondary" bold>
                        Alternatives
                      </Text>
                      <ul className="mt-1 grid gap-1">
                        {decision.alternatives.map((item) => (
                          <li key={item} className="text-xs text-kumo-subtle">
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              {(decision.relatedAssumptionIds.length > 0 ||
                decision.relatedPreprocessingStepIds.length > 0) && (
                <div className="flex flex-wrap gap-2">
                  {decision.relatedAssumptionIds.map((id) => (
                    <Badge key={`assumption-${id}`} variant="secondary">
                      {id}
                    </Badge>
                  ))}
                  {decision.relatedPreprocessingStepIds.map((id) => (
                    <Badge key={`step-${id}`} variant="secondary">
                      {id}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </details>
        ))}
      </div>
    </div>
  );
}

function ColumnList({ label, columns }: { label: string; columns: string[] }) {
  return (
    <div className="rounded-lg border border-kumo-line bg-kumo-base p-3">
      <Text size="xs" variant="secondary">
        {label}
      </Text>
      <Text size="sm" bold>
        {columns.length > 0 ? columns.join(", ") : "None"}
      </Text>
    </div>
  );
}

function DatasetIntentPanel({ intent }: { intent: DatasetIntent }) {
  return (
    <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
      <div className="mb-3">
        <Text size="sm" bold>
          Dataset intent
        </Text>
        <Text size="xs" variant="secondary">
          Accepted assumptions become the canonical state used by downstream
          planning.
        </Text>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <ColumnList
          label="Target"
          columns={intent.targetColumn ? [intent.targetColumn] : []}
        />
        <ColumnList label="Features" columns={intent.featureColumns} />
        <ColumnList label="Ignored" columns={intent.ignoredColumns} />
        <ColumnList label="Identifiers" columns={intent.identifierColumns} />
        <ColumnList label="Timestamps" columns={intent.timestampColumns} />
        <ColumnList label="Text" columns={intent.textColumns} />
      </div>

      {(intent.conflicts.length > 0 || intent.warnings.length > 0) && (
        <div className="mt-3 grid gap-2">
          {intent.conflicts.length > 0 && (
            <div className="rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 p-3">
              <Text size="xs" bold>
                Conflicts
              </Text>
              <ul className="mt-2 grid gap-1">
                {intent.conflicts.map((conflict) => (
                  <li key={conflict} className="text-xs text-kumo-subtle">
                    {conflict}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {intent.warnings.length > 0 && (
            <div className="rounded-lg border border-kumo-warning/40 bg-kumo-warning/10 p-3">
              <Text size="xs" bold>
                Intent warnings
              </Text>
              <ul className="mt-2 grid gap-1">
                {intent.warnings.map((warning) => (
                  <li key={warning} className="text-xs text-kumo-subtle">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AiReviewPanel({
  summary,
  reviewState,
  onGenerate,
  onUpdateAssumption,
  onSendToChat
}: {
  summary: DatasetSummary;
  reviewState: AiReviewState;
  onGenerate: () => void;
  onUpdateAssumption: (
    assumptionId: string,
    status: PreprocessingStatus
  ) => void;
  onSendToChat: () => void;
}) {
  const plan = reviewState.status === "ready" ? reviewState.plan : null;
  const groupedAssumptions = useMemo(() => {
    const groups = new Map<ColumnAssumption["role"], ColumnAssumption[]>();

    for (const assumption of plan?.assumptions ?? []) {
      const current = groups.get(assumption.role) ?? [];
      current.push(assumption);
      groups.set(assumption.role, current);
    }

    const order: ColumnAssumption["role"][] = [
      "target",
      "identifier",
      "timestamp",
      "free_text",
      "ignore",
      "feature",
      "unknown"
    ];

    return order
      .map((role) => [role, groups.get(role) ?? []] as const)
      .filter(([, assumptions]) => assumptions.length > 0);
  }, [plan]);
  const acceptedCount =
    plan?.assumptions.filter((assumption) => assumption.status === "accepted")
      .length ?? 0;
  const rejectedCount =
    plan?.assumptions.filter((assumption) => assumption.status === "rejected")
      .length ?? 0;

  return (
    <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <MagnifyingGlassIcon size={16} />
            <Text size="sm" bold>
              AI dataset review
            </Text>
          </div>
          <Text size="xs" variant="secondary">
            Sends profile statistics, samples, and preview rows only.
          </Text>
        </div>
        <div className="flex flex-wrap gap-2">
          {plan && (
            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={<ChatCircleDotsIcon size={14} />}
              onClick={onSendToChat}
            >
              Discuss
            </Button>
          )}
          <Button
            type="button"
            variant="primary"
            size="sm"
            icon={<MagnifyingGlassIcon size={14} />}
            disabled={reviewState.status === "loading"}
            onClick={onGenerate}
          >
            {reviewState.status === "loading" ? "Reviewing" : "Generate review"}
          </Button>
        </div>
      </div>

      {reviewState.status === "idle" && (
        <div className="mt-3 rounded-lg border border-dashed border-kumo-line bg-kumo-base px-3 py-4">
          <Text size="xs" variant="secondary">
            Generate a structured review for {summary.columns.length} columns
            before accepting preprocessing assumptions.
          </Text>
        </div>
      )}

      {reviewState.status === "error" && (
        <div className="mt-3 rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 px-3 py-2">
          <Text size="sm">{reviewState.message}</Text>
        </div>
      )}

      {plan && (
        <div className="mt-3 grid gap-3">
          <div className="rounded-lg border border-kumo-line bg-kumo-base p-3">
            <Text size="sm" bold>
              {plan.datasetSummary}
            </Text>
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge variant="secondary">
                {plan.assumptions.length} assumptions
              </Badge>
              <Badge variant="secondary">
                {plan.decisions.length} decisions
              </Badge>
              <Badge variant="secondary">{acceptedCount} accepted</Badge>
              <Badge variant="secondary">{rejectedCount} rejected</Badge>
            </div>
          </div>

          <DecisionTrace decisions={plan.decisions} />

          {plan.globalWarnings.length > 0 && (
            <div className="rounded-lg border border-kumo-warning/40 bg-kumo-warning/10 p-3">
              <Text size="xs" bold>
                Global warnings
              </Text>
              <ul className="mt-2 grid gap-1">
                {plan.globalWarnings.map((warning) => (
                  <li key={warning} className="text-xs text-kumo-subtle">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {groupedAssumptions.map(([role, assumptions]) => (
            <div key={role} className="grid gap-2">
              <Text size="xs" variant="secondary" bold>
                {assumptionRoleLabel(role)}
              </Text>
              {assumptions.map((assumption) => (
                <div
                  key={assumption.id}
                  className={`rounded-lg border p-3 ${
                    assumption.status === "accepted"
                      ? "border-kumo-brand bg-kumo-brand/5"
                      : assumption.status === "rejected"
                        ? "border-kumo-line bg-kumo-base opacity-70"
                        : "border-kumo-line bg-kumo-base"
                  }`}
                >
                  <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Text size="sm" bold>
                          {assumption.columnName}
                        </Text>
                        <Badge variant="secondary">
                          {assumption.semanticType}
                        </Badge>
                        <Badge
                          variant={
                            assumption.confidence === "high"
                              ? "primary"
                              : "secondary"
                          }
                        >
                          {assumption.confidence}
                        </Badge>
                        <Badge
                          variant={
                            assumption.status === "accepted"
                              ? "primary"
                              : "secondary"
                          }
                        >
                          {assumption.status}
                        </Badge>
                      </div>
                      {assumption.evidence.length > 0 && (
                        <Text size="xs" variant="secondary">
                          {assumption.evidence.join(" ")}
                        </Text>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="primary"
                        size="sm"
                        disabled={assumption.status === "accepted"}
                        onClick={() =>
                          onUpdateAssumption(assumption.id, "accepted")
                        }
                      >
                        Accept
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        disabled={assumption.status === "rejected"}
                        onClick={() =>
                          onUpdateAssumption(assumption.id, "rejected")
                        }
                      >
                        Reject
                      </Button>
                    </div>
                  </div>

                  {(assumption.risks.length > 0 ||
                    assumption.recommendedActions.length > 0) && (
                    <div className="mt-3 grid gap-2 sm:grid-cols-2">
                      {assumption.risks.length > 0 && (
                        <div>
                          <Text size="xs" variant="secondary" bold>
                            Risks
                          </Text>
                          <ul className="mt-1 grid gap-1">
                            {assumption.risks.map((risk) => (
                              <li
                                key={risk}
                                className="text-xs text-kumo-subtle"
                              >
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {assumption.recommendedActions.length > 0 && (
                        <div>
                          <Text size="xs" variant="secondary" bold>
                            Actions
                          </Text>
                          <ul className="mt-1 grid gap-1">
                            {assumption.recommendedActions.map((action) => (
                              <li
                                key={action}
                                className="text-xs text-kumo-subtle"
                              >
                                {action}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}

          {plan.nextQuestions.length > 0 && (
            <div className="rounded-lg border border-kumo-line bg-kumo-base p-3">
              <Text size="xs" variant="secondary" bold>
                Next questions
              </Text>
              <ul className="mt-2 grid gap-1">
                {plan.nextQuestions.map((question) => (
                  <li key={question} className="text-xs text-kumo-subtle">
                    {question}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SortableColumnHeader({
  label,
  sortKey,
  currentSort,
  onSort
}: {
  label: string;
  sortKey: ColumnSortKey;
  currentSort: ColumnSort;
  onSort: (key: ColumnSortKey) => void;
}) {
  const active = currentSort.key === sortKey;

  return (
    <th className="px-3 py-2 font-medium">
      <button
        type="button"
        className="inline-flex items-center gap-1 text-left text-kumo-subtle hover:text-kumo-default"
        onClick={() => onSort(sortKey)}
      >
        {label}
        {active &&
          (currentSort.direction === "asc" ? (
            <CaretUpIcon size={12} weight="bold" />
          ) : (
            <CaretDownIcon size={12} weight="bold" />
          ))}
      </button>
    </th>
  );
}

function DatasetWorkspace() {
  const toasts = useKumoToastManager();
  const csvInputRef = useRef<HTMLInputElement>(null);
  const [isCsvDragging, setIsCsvDragging] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>({
    status: "idle"
  });
  const [preprocessingStatuses, setPreprocessingStatuses] = useState<
    Record<string, PreprocessingStatus>
  >({});
  const [columnFilter, setColumnFilter] = useState<ColumnFilter>("all");
  const [columnSort, setColumnSort] = useState<ColumnSort>({
    key: "notesCount",
    direction: "desc"
  });
  const [aiReviewState, setAiReviewState] = useState<AiReviewState>({
    status: "idle"
  });
  const [intentOverrides, setIntentOverrides] = useState<IntentOverride[]>([]);

  const currentSummary =
    uploadState.status === "ready" ? uploadState.summary : null;
  const proposedPreprocessingSteps =
    currentSummary?.proposedPreprocessingSteps ?? [];
  const effectivePreprocessingSteps = proposedPreprocessingSteps.map((step) =>
    getEffectivePreprocessingStep(step, preprocessingStatuses)
  );
  const acceptedPreprocessingSteps = effectivePreprocessingSteps.filter(
    (step) => step.status === "accepted"
  );
  const transformedPreviewRows =
    currentSummary && acceptedPreprocessingSteps.length > 0
      ? applyPreprocessingStepsToPreviewRows(
          currentSummary.previewRows,
          acceptedPreprocessingSteps
        )
      : [];
  const columnQualityCounts = useMemo(() => {
    const columns = currentSummary?.columns ?? [];

    return {
      hasNotes: columns.filter((column) => column.profilingNotes.length > 0)
        .length,
      highMissingness: columns.filter((column) =>
        hasProfilingNote(column, "high_missingness")
      ).length,
      likelyIdentifiers: columns.filter((column) =>
        hasProfilingNote(column, "likely_identifier")
      ).length,
      empty: columns.filter((column) =>
        hasProfilingNote(column, "empty_column")
      ).length,
      constant: columns.filter((column) =>
        hasProfilingNote(column, "constant_column")
      ).length
    };
  }, [currentSummary]);
  const visibleColumns = useMemo(() => {
    if (!currentSummary) return [];
    const direction = columnSort.direction === "asc" ? 1 : -1;

    return [...currentSummary.columns]
      .filter((column) => filterColumn(column, columnFilter))
      .sort((left, right) => {
        const leftValue = getColumnSortValue(
          left,
          columnSort.key,
          currentSummary.parsedRowCount
        );
        const rightValue = getColumnSortValue(
          right,
          columnSort.key,
          currentSummary.parsedRowCount
        );

        if (typeof leftValue === "number" && typeof rightValue === "number") {
          if (leftValue !== rightValue)
            return (leftValue - rightValue) * direction;
        } else {
          const comparison = String(leftValue).localeCompare(
            String(rightValue)
          );
          if (comparison !== 0) return comparison * direction;
        }

        if (columnSort.key !== "notesCount") {
          const notesDelta =
            right.profilingNotes.length - left.profilingNotes.length;
          if (notesDelta !== 0) return notesDelta;
        }

        const missingDelta =
          getMissingPercent(right, currentSummary.parsedRowCount) -
          getMissingPercent(left, currentSummary.parsedRowCount);
        if (missingDelta !== 0) return missingDelta;

        return left.name.localeCompare(right.name);
      });
  }, [columnFilter, columnSort, currentSummary]);
  const datasetIntent = useMemo(() => {
    if (!currentSummary || aiReviewState.status !== "ready") return null;
    return buildDatasetIntent(
      currentSummary,
      aiReviewState.plan.assumptions,
      intentOverrides
    );
  }, [aiReviewState, currentSummary, intentOverrides]);

  const updatePreprocessingStatus = useCallback(
    (stepId: string, status: PreprocessingStatus) => {
      setPreprocessingStatuses((current) => ({
        ...current,
        [stepId]: status
      }));
    },
    []
  );
  const updateColumnSort = useCallback((key: ColumnSortKey) => {
    setColumnSort((current) => ({
      key,
      direction:
        current.key === key && current.direction === "desc" ? "asc" : "desc"
    }));
  }, []);

  const resetWorkspace = useCallback(() => {
    setUploadState({ status: "idle" });
    setPreprocessingStatuses({});
    setAiReviewState({ status: "idle" });
    setIntentOverrides([]);
    setColumnFilter("all");
    setColumnSort({ key: "notesCount", direction: "desc" });
    if (csvInputRef.current) csvInputRef.current.value = "";
  }, []);

  const generateAiReview = useCallback(async () => {
    if (!currentSummary) return;

    setAiReviewState({ status: "loading" });
    try {
      const response = await fetch("/api/ai-review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          profile: buildAiReviewProfile(currentSummary)
        })
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          (data as { error?: string } | null)?.error ||
            "The AI review could not be generated."
        );
      }

      setAiReviewState({
        status: "ready",
        plan: data as LlmPreprocessingPlan
      });
      setIntentOverrides([]);
      toasts.add({
        title: "AI review generated",
        description: "Review the proposed assumptions before using them."
      });
    } catch (error) {
      setAiReviewState({
        status: "error",
        message:
          error instanceof Error
            ? error.message
            : "The AI review could not be generated."
      });
    }
  }, [currentSummary, toasts]);

  const updateAiAssumptionStatus = useCallback(
    (assumptionId: string, status: PreprocessingStatus) => {
      setAiReviewState((current) => {
        if (current.status !== "ready") return current;
        const assumption = current.plan.assumptions.find(
          (item) => item.id === assumptionId
        );
        if (assumption) {
          setIntentOverrides((overrides) => [
            ...overrides.filter(
              (override) => override.assumptionId !== assumptionId
            ),
            buildIntentOverride(assumption, status)
          ]);
        }
        return {
          status: "ready",
          plan: updateAssumptionStatus(current.plan, assumptionId, status)
        };
      });
    },
    []
  );

  const sendAiReviewToChat = useCallback(() => {
    if (!currentSummary || aiReviewState.status !== "ready") return;

    window.dispatchEvent(
      new CustomEvent("automated-fe:agent-chat-message", {
        detail: {
          text: buildReviewChatMessage(currentSummary, aiReviewState.plan)
        }
      })
    );
    toasts.add({
      title: "Review sent to chat",
      description: "The agent sidebar now has the current review context."
    });
  }, [aiReviewState, currentSummary, toasts]);

  const handleCsvFile = useCallback(
    async (file: File) => {
      if (uploadState.status === "ready") {
        const replace = window.confirm(
          "Uploading a new CSV will replace the active dataset and preprocessing pipeline."
        );
        if (!replace) return;
        setPreprocessingStatuses({});
        setAiReviewState({ status: "idle" });
        setIntentOverrides([]);
      }

      setUploadState({ status: "validating", fileName: file.name });
      const validationError = validateCsvFile(file);
      if (validationError) {
        setUploadState({ status: "error", message: validationError });
        return;
      }

      setUploadState({ status: "parsing", fileName: file.name });
      try {
        const summary = await parseCsvFile(file);
        setUploadState({ status: "ready", summary });
        setPreprocessingStatuses({});
        setAiReviewState({ status: "idle" });
        setIntentOverrides([]);
        toasts.add({
          title: "CSV parsed",
          description: `${summary.parsedRowCount.toLocaleString()} rows, ${summary.columns.length} columns`
        });
      } catch (error) {
        setUploadState({
          status: "error",
          message:
            error instanceof Error
              ? error.message
              : "The CSV could not be parsed."
        });
      } finally {
        if (csvInputRef.current) csvInputRef.current.value = "";
      }
    },
    [toasts, uploadState.status]
  );

  const handleCsvInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) void handleCsvFile(file);
    },
    [handleCsvFile]
  );

  const handleCsvDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setIsCsvDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) void handleCsvFile(file);
    },
    [handleCsvFile]
  );

  return (
    <Surface
      className={`rounded-xl ring p-4 transition-colors ${
        isCsvDragging
          ? "ring-2 ring-kumo-brand bg-kumo-brand/5"
          : "ring-kumo-line"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer.types.includes("Files")) {
          setIsCsvDragging(true);
        }
      }}
      onDragLeave={(event) => {
        event.preventDefault();
        event.stopPropagation();
        if (event.currentTarget === event.target) {
          setIsCsvDragging(false);
        }
      }}
      onDrop={handleCsvDrop}
    >
      <input
        ref={csvInputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={handleCsvInputChange}
      />

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <Text size="lg" bold>
              Active dataset
            </Text>
            <Text size="sm" variant="secondary">
              Upload one CSV file to inspect columns and review preprocessing
              recommendations.
            </Text>
          </div>
          <div className="flex gap-2">
            {currentSummary && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                icon={<TrashIcon size={14} />}
                onClick={resetWorkspace}
              >
                Reset
              </Button>
            )}
            <Button
              type="button"
              variant="primary"
              size="sm"
              icon={<PaperclipIcon size={14} />}
              onClick={() => csvInputRef.current?.click()}
              disabled={
                uploadState.status === "validating" ||
                uploadState.status === "parsing"
              }
            >
              Choose CSV
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-dashed border-kumo-line bg-kumo-elevated px-4 py-5 text-center">
          <Text size="sm" bold>
            Drop a CSV here
          </Text>
          <Text size="xs" variant="secondary">
            `.csv` only, up to {formatBytes(MAX_CSV_SIZE_BYTES)}. Profiling runs
            locally in the browser.
          </Text>
        </div>

        {uploadState.status === "validating" && (
          <Badge variant="secondary">Validating {uploadState.fileName}</Badge>
        )}
        {uploadState.status === "parsing" && (
          <Badge variant="secondary">Parsing {uploadState.fileName}</Badge>
        )}
        {uploadState.status === "error" && (
          <div className="rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 px-3 py-2">
            <Text size="sm">{uploadState.message}</Text>
          </div>
        )}

        {currentSummary ? (
          <div className="grid gap-4">
            <div className="grid gap-2 sm:grid-cols-4">
              <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
                <Text size="xs" variant="secondary">
                  File
                </Text>
                <Text size="sm" bold>
                  {currentSummary.fileName}
                </Text>
              </div>
              <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
                <Text size="xs" variant="secondary">
                  Size
                </Text>
                <Text size="sm" bold>
                  {formatBytes(currentSummary.fileSizeBytes)}
                </Text>
              </div>
              <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
                <Text size="xs" variant="secondary">
                  Rows sampled
                </Text>
                <Text size="sm" bold>
                  {currentSummary.parsedRowCount.toLocaleString()}
                </Text>
              </div>
              <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
                <Text size="xs" variant="secondary">
                  Columns
                </Text>
                <Text size="sm" bold>
                  {currentSummary.columns.length}
                </Text>
              </div>
            </div>

            {currentSummary.warnings.length > 0 && (
              <div className="rounded-lg border border-kumo-warning/40 bg-kumo-warning/10 px-3 py-2">
                {currentSummary.warnings.map((warning) => (
                  <Text key={warning} size="xs">
                    {warning}
                  </Text>
                ))}
              </div>
            )}

            <AiReviewPanel
              summary={currentSummary}
              reviewState={aiReviewState}
              onGenerate={() => void generateAiReview()}
              onUpdateAssumption={updateAiAssumptionStatus}
              onSendToChat={sendAiReviewToChat}
            />

            {datasetIntent && <DatasetIntentPanel intent={datasetIntent} />}

            {proposedPreprocessingSteps.length > 0 && (
              <div className="rounded-lg border border-kumo-warning/40 bg-kumo-warning/10 p-3">
                <div className="mb-3">
                  <Text size="sm" bold>
                    Preprocessing recommendations
                  </Text>
                  <Text size="xs" variant="secondary">
                    Raw values are unchanged. Accepting a recommendation records
                    intent for the preprocessing pipeline.
                  </Text>
                </div>
                <div className="grid gap-2">
                  {effectivePreprocessingSteps.map((step) => {
                    const details = describePreprocessingStep(step);

                    return (
                      <div
                        key={step.id}
                        className="flex flex-col gap-2 rounded-lg border border-kumo-line bg-kumo-base p-3 sm:flex-row sm:items-center sm:justify-between"
                      >
                        <div>
                          <div className="flex flex-wrap items-center gap-2">
                            <Text size="sm" bold>
                              {details.title}
                            </Text>
                            <Badge
                              variant={
                                step.status === "accepted"
                                  ? "primary"
                                  : "secondary"
                              }
                            >
                              {step.status}
                            </Badge>
                          </div>
                          <Text size="xs" variant="secondary">
                            {details.description}
                          </Text>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            type="button"
                            variant="primary"
                            size="sm"
                            disabled={step.status === "accepted"}
                            onClick={() =>
                              updatePreprocessingStatus(step.id, "accepted")
                            }
                          >
                            Accept
                          </Button>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            disabled={step.status === "rejected"}
                            onClick={() =>
                              updatePreprocessingStatus(step.id, "rejected")
                            }
                          >
                            Reject
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-3">
              <div className="mb-3">
                <Text size="sm" bold>
                  Preprocessing pipeline
                </Text>
                <Text size="xs" variant="secondary">
                  Accepted steps are tracked separately from the raw dataset.
                </Text>
              </div>
              {acceptedPreprocessingSteps.length > 0 ? (
                <ol className="grid gap-2">
                  {acceptedPreprocessingSteps.map((step, index) => {
                    const details = describePreprocessingStep(step);

                    return (
                      <li
                        key={step.id}
                        className="flex items-start gap-2 rounded-lg border border-kumo-line bg-kumo-base p-2"
                      >
                        <Badge variant="primary">{index + 1}</Badge>
                        <div>
                          <Text size="sm" bold>
                            {details.title}
                          </Text>
                          <Text size="xs" variant="secondary">
                            {details.description}
                          </Text>
                        </div>
                      </li>
                    );
                  })}
                </ol>
              ) : (
                <Text size="xs" variant="secondary">
                  No preprocessing steps accepted yet.
                </Text>
              )}
            </div>

            <div className="rounded-lg border border-kumo-line bg-kumo-elevated">
              <div className="grid gap-3 border-b border-kumo-line p-3">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    {columnQualityCounts.hasNotes} columns with notes
                  </Badge>
                  <Badge variant="secondary">
                    {columnQualityCounts.highMissingness} high missingness
                  </Badge>
                  <Badge variant="secondary">
                    {columnQualityCounts.likelyIdentifiers} likely IDs
                  </Badge>
                  <Badge variant="secondary">
                    {columnQualityCounts.empty} empty
                  </Badge>
                  <Badge variant="secondary">
                    {columnQualityCounts.constant} constant
                  </Badge>
                </div>
                <div className="flex flex-wrap gap-2">
                  {COLUMN_FILTERS.map((filter) => (
                    <Button
                      key={filter.key}
                      type="button"
                      size="sm"
                      variant={
                        columnFilter === filter.key ? "primary" : "secondary"
                      }
                      onClick={() => setColumnFilter(filter.key)}
                    >
                      {filter.label}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="overflow-auto">
                <table className="min-w-full text-left text-sm">
                  <thead className="bg-kumo-elevated text-kumo-subtle">
                    <tr>
                      <SortableColumnHeader
                        label="Column"
                        sortKey="name"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Type"
                        sortKey="type"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Missing"
                        sortKey="missingCount"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="% missing"
                        sortKey="missingPercent"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Non-missing"
                        sortKey="nonMissingCount"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Unique"
                        sortKey="uniqueCount"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Unique ratio"
                        sortKey="uniqueRatio"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <th className="px-3 py-2 font-medium">Top values</th>
                      <th className="px-3 py-2 font-medium">Samples</th>
                      <SortableColumnHeader
                        label="Notes"
                        sortKey="notesCount"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-kumo-line">
                    {visibleColumns.map((column) => (
                      <tr key={column.name}>
                        <td className="px-3 py-2 font-medium text-kumo-default">
                          {column.name}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.inferredType}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.missingCount}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {formatPercent(
                            getMissingPercent(
                              column,
                              currentSummary.parsedRowCount
                            )
                          )}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.nonMissingCount.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.uniqueCount.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {formatPercent(column.uniqueRatio * 100)}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.topValues.length > 0 ? (
                            <div className="grid gap-1">
                              {column.topValues.map((topValue) => (
                                <span key={topValue.value}>
                                  {topValue.value} ({topValue.count},{" "}
                                  {formatPercent(topValue.percent)})
                                </span>
                              ))}
                            </div>
                          ) : (
                            "No non-empty values"
                          )}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.sampleValues.length > 0
                            ? column.sampleValues.join(", ")
                            : "No non-empty samples"}
                        </td>
                        <td className="px-3 py-2 text-kumo-subtle">
                          {column.profilingNotes.length > 0 ? (
                            <div className="grid gap-1">
                              {column.profilingNotes.map((note) => (
                                <span key={note.code}>
                                  {note.message} ({note.affectedCount})
                                </span>
                              ))}
                            </div>
                          ) : (
                            "No notes"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {visibleColumns.length === 0 && (
                  <div className="border-t border-kumo-line px-3 py-4">
                    <Text size="sm" variant="secondary">
                      No columns match the selected filter.
                    </Text>
                  </div>
                )}
              </div>
            </div>

            <details className="min-w-0 overflow-hidden rounded-lg border border-kumo-line bg-kumo-elevated">
              <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-kumo-default">
                Raw preview first {currentSummary.previewRows.length} rows
              </summary>
              <PreviewTable
                columns={currentSummary.columns}
                rows={currentSummary.previewRows}
              />
            </details>

            {acceptedPreprocessingSteps.length > 0 && (
              <details className="min-w-0 overflow-hidden rounded-lg border border-kumo-line bg-kumo-elevated">
                <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-kumo-default">
                  Transformed preview first {transformedPreviewRows.length} rows
                </summary>
                <PreviewTable
                  columns={currentSummary.columns}
                  rows={transformedPreviewRows}
                />
              </details>
            )}
          </div>
        ) : (
          <Empty
            icon={<CheckCircleIcon size={28} />}
            title="No dataset loaded"
            contents="Upload a CSV to inspect the schema and review preprocessing recommendations."
          />
        )}
      </div>
    </Surface>
  );
}

function AppShell() {
  return (
    <div className="min-h-screen bg-kumo-base text-kumo-default">
      <header className="border-b border-kumo-line bg-kumo-base">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div>
            <Text size="lg" bold>
              Automated FE
            </Text>
            <Text size="sm" variant="secondary">
              CSV profiling and preprocessing workspace
            </Text>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="min-w-0">
          <DatasetWorkspace />
        </div>
        <aside className="min-w-0">
          <AgentChatSidebar />
        </aside>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Toasty>
      <AppShell />
    </Toasty>
  );
}
