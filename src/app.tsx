import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart, type UIMessage } from "ai";
import type { ChatAgent } from "./server";
import type {
  FeatureValidationResult,
  ValidatedFeatureSuggestion
} from "./feature-engineering";
import {
  MAX_CSV_SIZE_BYTES,
  buildAiReviewProfile,
  formatBytes,
  isTrainingRow,
  parseCsvFile,
  renderPreviewValue,
  transformCsvFile,
  validateCsvFile,
  type ColumnSummary,
  type DatasetSummary,
  type ColumnDecision,
  type ColumnPreprocessingPlan,
  type ColumnSelectionPlan,
  type SelectedPreprocessingStep,
  type PreprocessingSuggestionAction,
  type ProfilingNoteCode,
  type SplitConfig
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
  DownloadSimpleIcon,
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
  | { status: "choosing_split"; fileName: string; file: File; rowCount: number }
  | { status: "parsing"; fileName: string }
  | {
      status: "ready";
      summary: DatasetSummary;
      file: File;
      splitConfig?: SplitConfig;
      rowCount: number;
    }
  | { status: "error"; message: string };

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
  | { status: "ready"; plan: ColumnSelectionPlan }
  | { status: "error"; message: string };

type PreprocessingReviewState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; plan: ColumnPreprocessingPlan }
  | { status: "error"; message: string };

type FeatureSuggestionState =
  | { status: "idle" }
  | { status: "loading" }
  | {
      status: "ready";
      result: FeatureValidationResult;
      decisions: FeatureSuggestionDecisions;
    }
  | { status: "error"; message: string };

type FeatureSuggestionDecision = "accepted" | "denied";

type FeatureSuggestionDecisions = Record<string, FeatureSuggestionDecision>;

type TransformState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "ready"; audit: string[]; rowCount: number; columns: number }
  | { status: "error"; message: string };

type ColumnPreparationAction = "feature" | "drop" | "review";

type PreprocessingChoice = PreprocessingSuggestionAction;

type PreprocessingChoices = Record<string, PreprocessingChoice[]>;

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

function countSplitRows(rowCount: number, seed: number, trainRatio: number) {
  let trainCount = 0;

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    if (isTrainingRow(rowIndex, seed, trainRatio)) {
      trainCount += 1;
    }
  }

  return {
    trainCount,
    testCount: rowCount - trainCount
  };
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
          <Text size="base" bold>
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
  const [datasetContext, setDatasetContext] = useState<string | null>(null);
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
      if (
        "addToolOutput" in event &&
        event.toolCall.toolName === "getActiveDatasetContext"
      ) {
        event.addToolOutput({
          toolCallId: event.toolCall.toolCallId,
          output: datasetContext
            ? { status: "ready", context: datasetContext }
            : { status: "empty", context: "No CSV is currently loaded." }
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
    function handleContext(event: Event) {
      const detail = (event as CustomEvent<{ context?: string | null }>).detail;
      setDatasetContext(detail?.context ?? null);
    }

    window.addEventListener("automated-fe:dataset-context", handleContext);
    return () =>
      window.removeEventListener("automated-fe:dataset-context", handleContext);
  }, []);

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
      className={`flex flex-col overflow-hidden rounded-2xl border border-kumo-line bg-kumo-elevated/50 backdrop-blur-sm transition-all shadow-sm lg:sticky lg:top-24 lg:h-[calc(100vh-8rem)] ${
        isDragging ? "ring-2 ring-kumo-brand bg-kumo-brand/5" : ""
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
      <div className="border-b border-kumo-line px-5 py-4 bg-kumo-base/30">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <ChatCircleDotsIcon size={20} className="text-kumo-brand" />
              <Text size="sm" bold>
                Agent Assistant
              </Text>
              <span
                className={`h-2 w-2 rounded-full ${
                  connected
                    ? "bg-kumo-success animate-pulse"
                    : "bg-kumo-inactive"
                }`}
                aria-label={connected ? "Connected" : "Disconnected"}
              />
            </div>
            <Text size="xs" variant="secondary" DANGEROUS_className="mt-0.5">
              Discuss preprocessing and data insights.
            </Text>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            shape="square"
            icon={<TrashIcon size={14} />}
            aria-label="Clear chat history"
            disabled={messages.length === 0}
            onClick={clearHistory}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-6 scrollbar-thin scrollbar-thumb-kumo-line scrollbar-track-transparent">
        {messages.length === 0 ? (
          <div className="flex flex-col h-full justify-center">
            <Empty
              icon={<BrainIcon size={32} className="text-kumo-brand/50" />}
              title="Start a conversation"
              contents="Ask about your dataset features, or request a summary of the current plan."
            />
            <div className="grid gap-2 mt-8">
              {promptSuggestions.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  disabled={!connected || isStreaming}
                  className="text-left px-4 py-2 rounded-xl border border-kumo-line bg-kumo-base hover:border-kumo-brand/40 hover:bg-kumo-brand/5 transition-all text-xs text-kumo-subtle hover:text-kumo-default"
                  onClick={() =>
                    sendMessage({
                      role: "user",
                      parts: [{ type: "text", text: prompt }]
                    })
                  }
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="grid gap-6">
            {messages.map((message: UIMessage, index) => {
              const isUser = message.role === "user";
              const isLastAssistant =
                message.role === "assistant" && index === messages.length - 1;

              return (
                <div key={message.id} className="grid gap-3">
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
                          className="rounded-xl border border-kumo-line bg-kumo-base/50"
                          open={!isDone}
                        >
                          <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-kumo-inactive hover:text-kumo-subtle transition-colors">
                            <BrainIcon size={12} />
                            Agent Thought Process
                            <span className="ml-auto opacity-50">
                              {isDone ? "Done" : "Thinking..."}
                            </span>
                          </summary>
                          <div className="px-4 pb-4 text-xs text-kumo-subtle leading-relaxed italic border-t border-kumo-line/30 pt-3">
                            {reasoning.text}
                          </div>
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
                          className="max-h-48 max-w-[85%] rounded-2xl border border-kumo-line shadow-sm object-contain"
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
                            <div className="max-w-[85%] rounded-2xl rounded-tr-none bg-kumo-brand px-4 py-2.5 text-sm leading-relaxed text-kumo-inverse shadow-sm">
                              {text}
                            </div>
                          ) : (
                            <div className="max-w-[90%] rounded-2xl rounded-tl-none border border-kumo-line bg-kumo-base px-1 py-1 text-sm leading-relaxed text-kumo-default shadow-sm overflow-hidden">
                              <Streamdown
                                className="sd-theme p-3"
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
        className="border-t border-kumo-line p-4 bg-kumo-base/30"
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
  return (
    <div className="w-full max-w-full overflow-x-auto rounded-b-xl scrollbar-thin scrollbar-thumb-kumo-line scrollbar-track-transparent">
      <table className="w-full min-w-[800px] text-left text-xs">
        <thead className="bg-kumo-base/50 text-kumo-subtle sticky top-0 backdrop-blur-sm">
          <tr>
            {columns.slice(0, 20).map((column) => (
              <th
                key={column.name}
                className="truncate px-4 py-3 font-semibold uppercase tracking-wider"
                title={column.name}
              >
                {column.name}
              </th>
            ))}
            {columns.length > 20 && (
              <th className="px-4 py-3 font-medium text-kumo-inactive italic">
                +{columns.length - 20} more columns
              </th>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-kumo-line/50">
          {rows.map((row, rowIndex) => (
            <tr
              key={rowIndex}
              className="hover:bg-kumo-base/30 transition-colors"
            >
              {columns.slice(0, 20).map((column) => (
                <td
                  key={column.name}
                  className="truncate px-4 py-3 text-kumo-subtle max-w-[200px]"
                  title={renderPreviewValue(row[column.name])}
                >
                  {renderPreviewValue(row[column.name])}
                </td>
              ))}
              {columns.length > 20 && (
                <td className="px-4 py-3 bg-kumo-base/5 opacity-50" />
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function buildReviewChatMessage(
  summary: DatasetSummary,
  columnPlan: ColumnSelectionPlan,
  preprocessingPlan: ColumnPreprocessingPlan | null
) {
  return `Use this compact dataset profile and current preparation plan as context for our conversation.

Dataset profile:
${JSON.stringify(buildAiReviewProfile(summary), null, 2)}

Column selection:
${JSON.stringify(
  {
    datasetSummary: columnPlan.datasetSummary,
    targetSuggestion: columnPlan.targetSuggestion,
    columnDecisions: columnPlan.columnDecisions,
    globalWarnings: columnPlan.globalWarnings,
    nextQuestions: columnPlan.nextQuestions
  },
  null,
  2
)}

Preprocessing:
${JSON.stringify(preprocessingPlan, null, 2)}

When suggesting changes, refer to column names and whether they should be kept, dropped, reviewed, or preprocessed.`;
}

function buildActiveDatasetContext({
  summary,
  targetColumn,
  columnActions,
  finalizedFeatureColumns,
  preprocessingChoices,
  preprocessingPlan
}: {
  summary: DatasetSummary;
  targetColumn: string | null;
  columnActions: Record<string, ColumnPreparationAction>;
  finalizedFeatureColumns: string[] | null;
  preprocessingChoices: PreprocessingChoices;
  preprocessingPlan: ColumnPreprocessingPlan | null;
}) {
  const featureColumns =
    finalizedFeatureColumns ??
    summary.columns
      .map((column) => column.name)
      .filter(
        (columnName) =>
          columnName !== targetColumn && columnActions[columnName] === "feature"
      );
  const droppedColumns = summary.columns
    .map((column) => column.name)
    .filter(
      (columnName) =>
        columnName !== targetColumn &&
        !featureColumns.includes(columnName) &&
        columnActions[columnName] === "drop"
    );

  return JSON.stringify(
    {
      dataset: buildAiReviewProfile(summary),
      targetColumn,
      featureColumns,
      droppedColumns,
      approvedPreprocessing:
        buildSelectedPreprocessingSteps(preprocessingChoices),
      preprocessingSuggestions:
        preprocessingPlan?.preprocessingSuggestions ?? []
    },
    null,
    2
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

function getDecisionForColumn(
  plan: ColumnSelectionPlan | null,
  columnName: string
) {
  return plan?.columnDecisions.find(
    (decision) => decision.columnName === columnName
  );
}

function getSuggestedTarget(plan: ColumnSelectionPlan | null) {
  return plan?.targetSuggestion ?? null;
}

function getColumnPreparationAction(
  column: ColumnSummary,
  decision: ColumnDecision | undefined,
  targetColumn: string | null
): ColumnPreparationAction {
  if (column.name === targetColumn) return "feature";
  if (decision?.decision === "drop") {
    return "drop";
  }
  if (decision?.decision === "review") {
    return "review";
  }
  return "feature";
}

function columnActionLabel(action: ColumnPreparationAction) {
  switch (action) {
    case "feature":
      return "Use";
    case "drop":
      return "Drop";
    case "review":
      return "Review";
  }
}

function preprocessingChoiceLabel(choice: PreprocessingChoice) {
  if (isNoPreprocessingChoice(choice)) return "No step";
  return choice;
}

function isNoPreprocessingChoice(choice: PreprocessingChoice) {
  const normalized = choice.trim().toLowerCase().replaceAll("_", " ");
  return normalized === "none" || normalized === "no step";
}

function isDropPreprocessingChoice(choice: PreprocessingChoice) {
  const normalized = choice.trim().toLowerCase().replaceAll("_", " ");
  return normalized === "drop" || normalized === "drop column";
}

function isLikelyNameColumn(column: ColumnSummary) {
  const name = column.name.toLowerCase();
  return (
    /(name|full_name|fullname|passengername|customer_name)/.test(name) ||
    column.sampleValues.some((value) =>
      /^[A-Z][a-z]+(?:\s+[A-Z][a-z.'-]+)+$/.test(value.trim())
    )
  );
}

function getSuggestedPreprocessingChoice(
  column: ColumnSummary,
  targetColumn: string | null
): { choice: PreprocessingChoice; reason: string } {
  const missingPercent =
    column.nonMissingCount + column.missingCount === 0
      ? 0
      : column.missingCount / (column.nonMissingCount + column.missingCount);

  if (column.name === targetColumn) {
    return {
      choice: "No step",
      reason: "Target column is kept separate from feature preprocessing."
    };
  }
  if (
    hasProfilingNote(column, "empty_column") ||
    hasProfilingNote(column, "constant_column")
  ) {
    return {
      choice: "Drop column",
      reason: "The column is unlikely to add useful feature signal."
    };
  }
  if (isLikelyNameColumn(column)) {
    return {
      choice: "Split full names into parts",
      reason:
        "Sample values look like full names, which can be split into reusable parts."
    };
  }
  if (hasProfilingNote(column, "missing_like_tokens")) {
    return {
      choice: "Standardize missing-value tokens",
      reason: "The profiler found tokens that commonly mean missing values."
    };
  }
  if (hasProfilingNote(column, "leading_trailing_whitespace")) {
    return {
      choice: "Trim whitespace",
      reason: "Sampled values contain leading or trailing whitespace."
    };
  }
  if (column.inferredType === "boolean") {
    return {
      choice: "Normalize boolean values",
      reason: "Boolean-like values should use one consistent representation."
    };
  }
  if (missingPercent > 0 && column.inferredType === "number") {
    return {
      choice:
        missingPercent >= 0.2
          ? "Fill missing values with the median"
          : "Fill missing values with the mean",
      reason:
        missingPercent >= 0.2
          ? "Numeric missingness is material; median is robust to skew."
          : "Numeric missingness is low; mean imputation is a simple baseline."
    };
  }
  if (missingPercent > 0 && column.inferredType === "string") {
    return {
      choice: "Fill missing values with the most common value",
      reason: "Categorical missing values can use the most frequent value."
    };
  }
  if (
    column.inferredType === "string" &&
    column.uniqueCount > 1 &&
    column.uniqueRatio <= 0.2
  ) {
    return {
      choice: "One-hot encode categories",
      reason: "Low-cardinality categorical values are suitable for encoding."
    };
  }
  return {
    choice: "No step",
    reason: "No required cleanup detected."
  };
}

function getPlanPreprocessingSuggestion(
  plan: ColumnPreprocessingPlan | null,
  column: ColumnSummary,
  targetColumn: string | null
) {
  const suggestion = plan?.preprocessingSuggestions.find(
    (item) => item.columnName === column.name
  );
  if (suggestion) {
    return {
      choice: suggestion.action,
      reason: suggestion.reason,
      implementation: suggestion.implementation,
      options: Array.from(
        new Set([suggestion.action, ...suggestion.alternatives])
      )
    };
  }

  const fallback = getSuggestedPreprocessingChoice(column, targetColumn);
  return {
    ...fallback,
    implementation: isNoPreprocessingChoice(fallback.choice)
      ? "Leave the column unchanged."
      : `${fallback.choice} for this column.`,
    options: [fallback.choice]
  };
}

function getSelectablePreprocessingOptions(
  suggestion: ReturnType<typeof getPlanPreprocessingSuggestion>,
  column: ColumnSummary
) {
  const options = new Set<PreprocessingChoice>([
    suggestion.choice,
    ...suggestion.options,
    "No step"
  ]);

  if (column.inferredType === "string") {
    options.add("Trim whitespace");
    options.add("Lowercase text");
  }
  if (isLikelyNameColumn(column)) {
    options.add("Split full names into parts");
  }
  if (column.inferredType === "boolean") {
    options.add("Normalize boolean values");
  }
  if (hasProfilingNote(column, "missing_like_tokens")) {
    options.add("Standardize missing-value tokens");
  }
  if (column.inferredType === "string" && column.uniqueRatio <= 0.2) {
    options.add("One-hot encode categories");
  }

  return Array.from(options);
}

function selectedNonNoStepChoices(choices: PreprocessingChoice[]) {
  return choices.filter((choice) => !isNoPreprocessingChoice(choice));
}

function getFeatureSuggestionKey(
  feature: ValidatedFeatureSuggestion,
  index: number
) {
  return `${feature.name}:${index}`;
}

function getAcceptedFeatureSuggestions(
  state: FeatureSuggestionState
): ValidatedFeatureSuggestion[] {
  if (state.status !== "ready") return [];

  return state.result.accepted.filter(
    (feature, index) =>
      state.decisions[getFeatureSuggestionKey(feature, index)] === "accepted"
  );
}

function buildSelectedPreprocessingSteps(
  choices: PreprocessingChoices
): SelectedPreprocessingStep[] {
  return Object.entries(choices).flatMap(([columnName, columnChoices]) =>
    selectedNonNoStepChoices(columnChoices).map((action) => ({
      columnName,
      action
    }))
  );
}

function SelectBox({
  value,
  onChange,
  children,
  ariaLabel,
  disabled = false
}: {
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  ariaLabel: string;
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      aria-label={ariaLabel}
      disabled={disabled}
      className="h-8 rounded-md border border-kumo-line bg-kumo-base px-2 text-sm text-kumo-default outline-none focus:ring-2 focus:ring-kumo-ring disabled:cursor-not-allowed disabled:opacity-60"
      onChange={(event) => onChange(event.target.value)}
    >
      {children}
    </select>
  );
}

function CollapsibleSection({
  title,
  subtitle,
  defaultOpen = true,
  children
}: {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="overflow-hidden rounded-xl border border-kumo-line bg-kumo-elevated shadow-sm">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-4 border-b border-kumo-line bg-kumo-base/30 p-5 text-left"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
      >
        <div>
          <Text size="base" bold>
            {title}
          </Text>
          {subtitle && (
            <Text size="xs" variant="secondary">
              {subtitle}
            </Text>
          )}
        </div>

        {open ? (
          <CaretUpIcon size={18} className="text-kumo-subtle" />
        ) : (
          <CaretDownIcon size={18} className="text-kumo-subtle" />
        )}
      </button>

      {open && <div>{children}</div>}
    </div>
  );
}

function PreparationReviewPanel({
  summary,
  reviewState,
  preprocessingState,
  targetColumn,
  columnActions,
  finalizedFeatureColumns,
  preprocessingChoices,
  transformState,
  onGenerate,
  onGeneratePreprocessing,
  featureSuggestionState,
  onGenerateFeatures,
  onFeatureSuggestionDecision,
  onDownloadTransformed,
  onAcceptTarget,
  onTargetChange,
  onFinishSelection,
  onColumnActionChange,
  onPreprocessingChoiceChange,
  onSendToChat
}: {
  summary: DatasetSummary;
  reviewState: AiReviewState;
  preprocessingState: PreprocessingReviewState;
  targetColumn: string | null;
  columnActions: Record<string, ColumnPreparationAction>;
  finalizedFeatureColumns: string[] | null;
  preprocessingChoices: PreprocessingChoices;
  transformState: TransformState;
  onGenerate: () => void;
  onGeneratePreprocessing: () => void;
  featureSuggestionState: FeatureSuggestionState;
  onGenerateFeatures: () => void;
  onFeatureSuggestionDecision: (
    featureKey: string,
    decision: FeatureSuggestionDecision
  ) => void;
  onDownloadTransformed: () => void;
  onAcceptTarget: (columnName: string) => void;
  onTargetChange: (columnName: string) => void;
  onFinishSelection: () => void;
  onColumnActionChange: (
    columnName: string,
    action: ColumnPreparationAction
  ) => void;
  onPreprocessingChoiceChange: (
    columnName: string,
    choice: PreprocessingChoice,
    checked: boolean
  ) => void;
  onSendToChat: () => void;
}) {
  const plan = reviewState.status === "ready" ? reviewState.plan : null;
  const preprocessingPlan =
    preprocessingState.status === "ready" ? preprocessingState.plan : null;
  const suggestedTarget = getSuggestedTarget(plan);
  const selectedTarget = targetColumn ?? suggestedTarget?.columnName ?? "";
  const targetEvidence = suggestedTarget?.reason || "";
  const finalizedFeatureSet = new Set(finalizedFeatureColumns ?? []);
  const isSelectionFinalized = finalizedFeatureColumns !== null;
  const preparedColumns = summary.columns.map((column) => {
    const decision = getDecisionForColumn(plan, column.name);
    const suggestedAction = getColumnPreparationAction(
      column,
      decision,
      targetColumn
    );
    const selectedAction =
      column.name === targetColumn
        ? "feature"
        : isSelectionFinalized
          ? finalizedFeatureSet.has(column.name)
            ? "feature"
            : "drop"
          : (columnActions[column.name] ?? suggestedAction);
    const suggestedPreprocessing = getPlanPreprocessingSuggestion(
      preprocessingPlan,
      column,
      targetColumn
    );
    const selectedPreprocessing =
      preprocessingChoices[column.name] ??
      (isNoPreprocessingChoice(suggestedPreprocessing.choice)
        ? []
        : [suggestedPreprocessing.choice]);

    return {
      column,
      decision,
      suggestedAction,
      selectedAction,
      suggestedPreprocessing,
      selectedPreprocessing
    };
  });
  const featureColumns = preparedColumns.filter(
    (item) =>
      item.column.name !== targetColumn && item.selectedAction === "feature"
  );
  const droppedColumns = preparedColumns.filter(
    (item) =>
      item.column.name !== targetColumn && item.selectedAction === "drop"
  );
  const reviewColumns = preparedColumns.filter(
    (item) =>
      item.column.name !== targetColumn && item.selectedAction === "review"
  );
  const activePreprocessing = preparedColumns.filter(
    (item) =>
      item.column.name !== targetColumn &&
      item.selectedAction === "feature" &&
      selectedNonNoStepChoices(item.selectedPreprocessing).length > 0
  );
  const acceptedEngineeredFeatures = getAcceptedFeatureSuggestions(
    featureSuggestionState
  );
  const featureDecisionCounts =
    featureSuggestionState.status === "ready"
      ? featureSuggestionState.result.accepted.reduce(
          (counts, feature, index) => {
            const decision =
              featureSuggestionState.decisions[
                getFeatureSuggestionKey(feature, index)
              ];
            if (decision === "accepted") counts.accepted += 1;
            else if (decision === "denied") counts.denied += 1;
            else counts.pending += 1;
            return counts;
          },
          { accepted: 0, denied: 0, pending: 0 }
        )
      : { accepted: 0, denied: 0, pending: 0 };
  const workflowSteps = [
    {
      label: "1. Target",
      value: targetColumn ?? "Choose outcome",
      complete: Boolean(targetColumn)
    },
    {
      label: "2. Columns",
      value: isSelectionFinalized
        ? `${featureColumns.length} kept`
        : `${reviewColumns.length} review`,
      complete: isSelectionFinalized
    },
    {
      label: "3. Preprocessing",
      value:
        preprocessingState.status === "ready"
          ? `${activePreprocessing.length} steps`
          : "Not generated",
      complete: preprocessingState.status === "ready"
    }
  ];

  return (
    <div className="grid gap-6">
      {plan && (
        <div className="grid gap-4 sm:grid-cols-3">
          {workflowSteps.map((step, idx) => (
            <div
              key={step.label}
              className={`relative overflow-hidden rounded-xl border p-4 transition-all ${
                step.complete
                  ? "border-kumo-brand/30 bg-kumo-brand/5 ring-1 ring-kumo-brand/20"
                  : "border-kumo-line bg-kumo-elevated opacity-80"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold ${step.complete ? "bg-kumo-brand text-kumo-inverse" : "bg-kumo-line text-kumo-inactive"}`}
                  >
                    {idx + 1}
                  </span>
                  <Text
                    size="xs"
                    variant="secondary"
                    bold
                    DANGEROUS_className="uppercase tracking-widest"
                  >
                    {step.label.split(". ")[1]}
                  </Text>
                </div>
                {step.complete && (
                  <CheckCircleIcon size={16} className="text-kumo-brand" />
                )}
              </div>
              <div className="mt-2">
                <Text
                  size="base"
                  bold
                  DANGEROUS_className={
                    step.complete ? "text-kumo-default" : "text-kumo-subtle"
                  }
                >
                  {step.value}
                </Text>
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="rounded-xl border border-kumo-line bg-kumo-elevated p-6 shadow-sm">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-3">
            <div className="mt-1 rounded-lg bg-kumo-brand/10 p-2 text-kumo-brand">
              <MagnifyingGlassIcon size={20} />
            </div>
            <div>
              <Text size="base" bold>
                1. Target Definition
              </Text>
              <Text
                size="sm"
                variant="secondary"
                DANGEROUS_className="max-w-xl"
              >
                {plan
                  ? plan.datasetSummary
                  : "Analyze your dataset to automatically identify the most likely prediction target."}
              </Text>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {plan && (
              <Button
                type="button"
                variant="secondary"
                size="sm"
                icon={<ChatCircleDotsIcon size={14} />}
                onClick={onSendToChat}
              >
                Discuss with Agent
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
              {reviewState.status === "loading"
                ? "Analyzing..."
                : "Suggest Plan"}
            </Button>
          </div>
        </div>

        {reviewState.status === "error" && (
          <div className="mt-4 rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 px-4 py-3">
            <Text size="sm" DANGEROUS_className="text-kumo-danger">
              {reviewState.message}
            </Text>
          </div>
        )}

        {plan && (
          <div className="mt-6 grid gap-4 rounded-xl border border-kumo-line bg-kumo-base p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <Text
                  size="xs"
                  variant="secondary"
                  bold
                  DANGEROUS_className="uppercase tracking-wider"
                >
                  Recommended Target
                </Text>
                <div className="mt-1 flex flex-wrap items-center gap-2">
                  <Text size="lg" bold>
                    {suggestedTarget?.columnName ?? "No target suggested"}
                  </Text>
                  {suggestedTarget && (
                    <Badge
                      variant={
                        suggestedTarget.confidence === "high"
                          ? "primary"
                          : "secondary"
                      }
                    >
                      {suggestedTarget.confidence} confidence
                    </Badge>
                  )}
                </div>
                {targetEvidence && (
                  <Text
                    size="sm"
                    variant="secondary"
                    DANGEROUS_className="mt-2 italic"
                  >
                    &quot;{targetEvidence}&quot;
                  </Text>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  disabled={!suggestedTarget}
                  onClick={() => {
                    if (suggestedTarget)
                      onAcceptTarget(suggestedTarget.columnName);
                  }}
                >
                  Apply Recommendation
                </Button>
                <div className="flex items-center gap-2">
                  <Text size="xs" variant="secondary">
                    or select
                  </Text>
                  <SelectBox
                    value={selectedTarget}
                    ariaLabel="Choose target column"
                    onChange={onTargetChange}
                  >
                    <option value="">Manual Selection</option>
                    {summary.columns.map((column) => (
                      <option key={column.name} value={column.name}>
                        {column.name}
                      </option>
                    ))}
                  </SelectBox>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {plan && (
        <>
          <div className="overflow-hidden rounded-xl border border-kumo-line bg-kumo-elevated shadow-sm">
            <div className="flex flex-col gap-3 border-b border-kumo-line bg-kumo-base/30 p-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-kumo-brand/10 p-2 text-kumo-brand">
                  <GearIcon size={20} />
                </div>
                <div>
                  <Text size="base" bold>
                    2. Feature Selection
                  </Text>
                  <Text size="sm" variant="secondary">
                    {featureColumns.length} kept, {droppedColumns.length}{" "}
                    dropped
                    {reviewColumns.length > 0
                      ? `, ${reviewColumns.length} pending review`
                      : ""}
                  </Text>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {plan.globalWarnings.map((warning) => (
                  <Badge
                    key={warning}
                    variant="secondary"
                    className="bg-kumo-warning/10 text-kumo-warning border-kumo-warning/20"
                  >
                    {warning}
                  </Badge>
                ))}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="bg-kumo-base/50 text-kumo-subtle">
                  <tr>
                    <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                      Column
                    </th>
                    <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                      AI Suggestion
                    </th>
                    <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                      Logic/Reason
                    </th>
                    <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                      Final Action
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-kumo-line/50">
                  {preparedColumns.map(
                    ({ column, decision, suggestedAction, selectedAction }) => (
                      <tr
                        key={column.name}
                        className="hover:bg-kumo-base/20 transition-colors"
                      >
                        <td className="px-4 py-3">
                          <div className="flex flex-col">
                            <Text size="sm" bold>
                              {column.name}
                            </Text>
                            <Text size="xs" variant="secondary">
                              {column.inferredType} ·{" "}
                              {formatPercent(
                                getMissingPercent(
                                  column,
                                  summary.parsedRowCount
                                )
                              )}{" "}
                              missing
                            </Text>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-wrap gap-2">
                            <Badge
                              variant={
                                suggestedAction === "drop"
                                  ? "secondary"
                                  : "primary"
                              }
                            >
                              {columnActionLabel(suggestedAction)}
                            </Badge>
                            {decision && (
                              <Badge variant="secondary" className="opacity-70">
                                {decision.confidence}
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="max-w-md px-4 py-3 text-xs text-kumo-subtle leading-relaxed">
                          {decision?.reason ??
                            column.profilingNotes[0]?.message ??
                            "No significant profiling notes."}
                        </td>
                        <td className="px-4 py-3">
                          <SelectBox
                            value={selectedAction}
                            ariaLabel={`Set role for ${column.name}`}
                            disabled={
                              isSelectionFinalized ||
                              column.name === targetColumn
                            }
                            onChange={(value) =>
                              onColumnActionChange(
                                column.name,
                                value as ColumnPreparationAction
                              )
                            }
                          >
                            <option value="feature">Use</option>
                            <option value="drop">Drop</option>
                            <option value="review">Review</option>
                          </SelectBox>
                        </td>
                      </tr>
                    )
                  )}
                </tbody>
              </table>
            </div>
            <div className="flex flex-col gap-3 border-t border-kumo-line bg-kumo-base/30 p-5 sm:flex-row sm:items-center sm:justify-between">
              <Text
                size="xs"
                variant="secondary"
                DANGEROUS_className="max-w-md"
              >
                {isSelectionFinalized
                  ? "Selection finalized. These features are now locked for preprocessing analysis."
                  : reviewColumns.length > 0
                    ? "Please resolve pending reviews before moving to preprocessing."
                    : "Finalize your selection to proceed with preprocessing recommendations."}
              </Text>
              <div className="flex flex-wrap gap-2 sm:gap-3">
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  icon={<CheckCircleIcon size={14} />}
                  disabled={
                    !targetColumn ||
                    isSelectionFinalized ||
                    reviewColumns.length > 0
                  }
                  onClick={onFinishSelection}
                >
                  Lock Selection
                </Button>
              </div>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-kumo-line bg-kumo-elevated shadow-sm">
            <div className="flex flex-col gap-3 border-b border-kumo-line bg-kumo-base/30 p-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-kumo-brand/10 p-2 text-kumo-brand">
                  <BrainIcon size={20} />
                </div>
                <div>
                  <Text size="base" bold>
                    3. Intelligent Preprocessing
                  </Text>
                  <Text size="sm" variant="secondary">
                    Dataset-aware transformations suggested by the model.
                  </Text>
                </div>
              </div>
              <Button
                type="button"
                variant="primary"
                size="sm"
                icon={<MagnifyingGlassIcon size={14} />}
                disabled={
                  !targetColumn ||
                  !isSelectionFinalized ||
                  preprocessingState.status === "loading"
                }
                onClick={onGeneratePreprocessing}
              >
                {preprocessingState.status === "loading"
                  ? "Analyzing Data..."
                  : "Generate Suggestions"}
              </Button>
            </div>
            {preprocessingState.status === "error" && (
              <div className="border-b border-kumo-line px-5 py-3">
                <Text size="sm" DANGEROUS_className="text-kumo-danger">
                  {preprocessingState.message}
                </Text>
              </div>
            )}
            {preprocessingState.status !== "ready" ? (
              <div className="px-5 py-10 text-center">
                <Text
                  size="sm"
                  variant="secondary"
                  DANGEROUS_className="max-w-sm mx-auto"
                >
                  Finalize Step 2 to enable preprocessing recommendations for
                  your kept features.
                </Text>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full text-left text-sm">
                  <thead className="bg-kumo-base/50 text-kumo-subtle">
                    <tr>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Column
                      </th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Recommended Step
                      </th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Context
                      </th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Selection
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-kumo-line/50">
                    {preparedColumns.map(
                      ({
                        column,
                        selectedAction,
                        suggestedPreprocessing,
                        selectedPreprocessing
                      }) =>
                        column.name !== targetColumn &&
                        selectedAction === "feature" ? (
                          <tr
                            key={column.name}
                            className="hover:bg-kumo-base/20 transition-colors"
                          >
                            <td className="px-4 py-3 font-bold text-kumo-default">
                              {column.name}
                            </td>
                            <td className="px-4 py-3">
                              <Badge
                                variant={
                                  isNoPreprocessingChoice(
                                    suggestedPreprocessing.choice
                                  )
                                    ? "secondary"
                                    : "primary"
                                }
                              >
                                {preprocessingChoiceLabel(
                                  suggestedPreprocessing.choice
                                )}
                              </Badge>
                            </td>
                            <td className="max-w-md px-4 py-3">
                              <div className="flex flex-col gap-1 text-xs text-kumo-subtle leading-relaxed">
                                <span>{suggestedPreprocessing.reason}</span>
                                {suggestedPreprocessing.implementation && (
                                  <span className="opacity-70 font-mono text-[10px]">
                                    {suggestedPreprocessing.implementation}
                                  </span>
                                )}
                              </div>
                            </td>
                            <td className="px-4 py-3">
                              <div className="grid min-w-[240px] gap-1.5">
                                {getSelectablePreprocessingOptions(
                                  suggestedPreprocessing,
                                  column
                                ).map((choice) => {
                                  const checked =
                                    !isNoPreprocessingChoice(choice) &&
                                    selectedPreprocessing.includes(choice);
                                  return (
                                    <label
                                      key={choice}
                                      className={`flex items-center gap-2 px-2 py-1 rounded transition-colors cursor-pointer ${checked ? "bg-kumo-brand/10 text-kumo-brand" : "hover:bg-kumo-base text-kumo-subtle"}`}
                                    >
                                      <input
                                        type="checkbox"
                                        checked={checked}
                                        disabled={isNoPreprocessingChoice(
                                          choice
                                        )}
                                        className="accent-kumo-brand"
                                        onChange={(event) =>
                                          onPreprocessingChoiceChange(
                                            column.name,
                                            choice,
                                            event.target.checked
                                          )
                                        }
                                      />
                                      <span className="text-[11px] font-medium">
                                        {preprocessingChoiceLabel(choice)}
                                      </span>
                                    </label>
                                  );
                                })}
                              </div>
                            </td>
                          </tr>
                        ) : null
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="overflow-hidden rounded-xl border border-kumo-line bg-kumo-elevated shadow-sm">
            <div className="flex flex-col gap-3 border-b border-kumo-line bg-kumo-base/30 p-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-kumo-brand/10 p-2 text-kumo-brand">
                  <BrainIcon size={20} />
                </div>
                <div>
                  <Text size="base" bold>
                    4. Feature Suggestions
                  </Text>
                  <Text size="sm" variant="secondary">
                    Candidate engineered features proposed for kept columns.
                  </Text>
                </div>
              </div>
              <Button
                type="button"
                variant="primary"
                size="sm"
                icon={<MagnifyingGlassIcon size={14} />}
                disabled={
                  !targetColumn ||
                  !isSelectionFinalized ||
                  preprocessingState.status !== "ready" ||
                  featureSuggestionState.status === "loading"
                }
                onClick={onGenerateFeatures}
              >
                {featureSuggestionState.status === "loading"
                  ? "Suggesting..."
                  : "Generate Features"}
              </Button>
            </div>
            <div className="p-5">
              {!isSelectionFinalized ? (
                <Text size="sm" variant="secondary">
                  Finalize Step 2 before generating engineered features.
                </Text>
              ) : preprocessingState.status !== "ready" ? (
                <Text size="sm" variant="secondary">
                  Generate and lock preprocessing choices before requesting
                  engineered features.
                </Text>
              ) : featureSuggestionState.status === "idle" ? (
                <Text size="sm" variant="secondary">
                  Click "Generate Features" to use the locked preprocessing
                  choices and prepared feature schema.
                </Text>
              ) : featureSuggestionState.status === "loading" ? (
                <Text size="sm" variant="secondary">
                  Generating feature suggestions...
                </Text>
              ) : featureSuggestionState.status === "error" ? (
                <div className="rounded-md border border-kumo-danger/40 bg-kumo-danger/10 px-4 py-3">
                  <Text size="sm" DANGEROUS_className="text-kumo-danger">
                    {featureSuggestionState.message}
                  </Text>
                </div>
              ) : (
                <div className="grid gap-4">
                  {featureSuggestionState.result.accepted.length > 0 && (
                    <div>
                      <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex items-center gap-2">
                          <CheckCircleIcon
                            size={16}
                            className="text-kumo-success"
                          />
                          <Text size="sm" bold>
                            Valid suggestions (
                            {featureSuggestionState.result.accepted.length})
                          </Text>
                        </div>
                        <Text size="xs" variant="secondary">
                          {featureDecisionCounts.accepted} accepted,{" "}
                          {featureDecisionCounts.denied} denied,{" "}
                          {featureDecisionCounts.pending} pending
                        </Text>
                      </div>
                      <div className="grid gap-2">
                        {featureSuggestionState.result.accepted.map(
                          (feature, idx) => {
                            const featureKey = getFeatureSuggestionKey(
                              feature,
                              idx
                            );
                            const decision =
                              featureSuggestionState.decisions[featureKey];
                            const isAccepted = decision === "accepted";
                            const isDenied = decision === "denied";

                            return (
                              <div
                                key={featureKey}
                                className={`rounded-lg border p-3 ${
                                  isAccepted
                                    ? "border-kumo-success/30 bg-kumo-success/5"
                                    : isDenied
                                      ? "border-kumo-danger/30 bg-kumo-danger/5"
                                      : "border-kumo-line bg-kumo-base/50"
                                }`}
                              >
                                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                                  <div className="flex-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <Text
                                        size="sm"
                                        bold
                                        DANGEROUS_className="text-kumo-default"
                                      >
                                        {feature.name}
                                      </Text>
                                      <Badge
                                        variant="secondary"
                                        className={`text-[10px] ${
                                          isAccepted
                                            ? "border-kumo-success/20 bg-kumo-success/10 text-kumo-success"
                                            : isDenied
                                              ? "border-kumo-danger/20 bg-kumo-danger/10 text-kumo-danger"
                                              : ""
                                        }`}
                                      >
                                        {isAccepted
                                          ? "Accepted"
                                          : isDenied
                                            ? "Denied"
                                            : "Pending"}
                                      </Badge>
                                    </div>
                                    <Text
                                      size="xs"
                                      variant="secondary"
                                      DANGEROUS_className="mt-1"
                                    >
                                      {feature.reason}
                                    </Text>
                                    <Text
                                      size="xs"
                                      DANGEROUS_className="mt-1 text-kumo-brand"
                                    >
                                      Expected benefit:{" "}
                                      {feature.expectedBenefit}
                                    </Text>
                                    {feature.warnings.length > 0 && (
                                      <div className="mt-2 flex flex-wrap gap-1">
                                        {feature.warnings.map(
                                          (warning, warningIndex) => (
                                            <Badge
                                              key={`${feature.name}-${warningIndex}`}
                                              variant="secondary"
                                              className="bg-kumo-warning/10 text-kumo-warning border-kumo-warning/20 text-[10px]"
                                            >
                                              {warning}
                                            </Badge>
                                          )
                                        )}
                                      </div>
                                    )}
                                  </div>
                                  <div className="flex flex-shrink-0 gap-2">
                                    <Button
                                      type="button"
                                      variant={
                                        isAccepted ? "primary" : "secondary"
                                      }
                                      size="sm"
                                      icon={<CheckCircleIcon size={14} />}
                                      disabled={isAccepted}
                                      onClick={() =>
                                        onFeatureSuggestionDecision(
                                          featureKey,
                                          "accepted"
                                        )
                                      }
                                    >
                                      Accept
                                    </Button>
                                    <Button
                                      type="button"
                                      variant="secondary"
                                      size="sm"
                                      icon={<XCircleIcon size={14} />}
                                      disabled={isDenied}
                                      onClick={() =>
                                        onFeatureSuggestionDecision(
                                          featureKey,
                                          "denied"
                                        )
                                      }
                                    >
                                      Deny
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            );
                          }
                        )}
                      </div>
                    </div>
                  )}
                  {featureSuggestionState.result.rejected.length > 0 && (
                    <div>
                      <div className="mb-3 flex items-center gap-2">
                        <XCircleIcon size={16} className="text-kumo-danger" />
                        <Text size="sm" bold>
                          Rejected (
                          {featureSuggestionState.result.rejected.length})
                        </Text>
                      </div>
                      <div className="grid gap-2">
                        {featureSuggestionState.result.rejected.map(
                          (rejection, idx) => (
                            <div
                              key={`rejected-${idx}`}
                              className="rounded-lg border border-kumo-danger/30 bg-kumo-danger/5 p-3"
                            >
                              <div className="flex items-start gap-2">
                                <div className="flex-1">
                                  <Text
                                    size="xs"
                                    bold
                                    DANGEROUS_className="text-kumo-danger"
                                  >
                                    Rejected
                                  </Text>
                                  <Text
                                    size="xs"
                                    variant="secondary"
                                    DANGEROUS_className="mt-1"
                                  >
                                    {rejection.reason}
                                  </Text>
                                </div>
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
                  {featureSuggestionState.result.accepted.length === 0 &&
                    featureSuggestionState.result.rejected.length === 0 && (
                      <Text size="sm" variant="secondary">
                        No feature suggestions were returned.
                      </Text>
                    )}
                </div>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-kumo-line bg-kumo-elevated p-6 shadow-sm">
            <div className="flex items-center gap-3 mb-6">
              <div className="rounded-lg bg-kumo-brand/10 p-2 text-kumo-brand">
                <CheckCircleIcon size={20} />
              </div>
              <Text size="base" bold>
                Final Preparation Plan
              </Text>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <ColumnList
                label="Prediction Target"
                columns={targetColumn ? [targetColumn] : []}
              />
              <ColumnList
                label="Feature Set"
                columns={featureColumns.map((item) => item.column.name)}
              />
              <ColumnList
                label="Excluded"
                columns={droppedColumns.map((item) => item.column.name)}
              />
            </div>

            <div className="mt-4 rounded-xl border border-kumo-line bg-kumo-base p-5">
              <div className="flex items-center gap-2 mb-3">
                <Badge
                  variant="secondary"
                  className="text-[10px] font-bold uppercase tracking-widest"
                >
                  Active Transforms
                </Badge>
              </div>
              {activePreprocessing.length > 0 ? (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {activePreprocessing.map((item) => (
                    <div
                      key={item.column.name}
                      className="flex flex-col gap-1 border-l-2 border-kumo-brand/30 pl-3"
                    >
                      <Text size="xs" bold DANGEROUS_className="truncate">
                        {item.column.name}
                      </Text>
                      <Text size="xs" variant="secondary">
                        {selectedNonNoStepChoices(item.selectedPreprocessing)
                          .map(preprocessingChoiceLabel)
                          .join(", ")}
                      </Text>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 text-kumo-inactive">
                  <Text size="sm">No custom transformations selected.</Text>
                </div>
              )}
              {acceptedEngineeredFeatures.length > 0 && (
                <div className="mt-4 border-t border-kumo-line pt-4">
                  <Text
                    size="xs"
                    bold
                    variant="secondary"
                    DANGEROUS_className="mb-3 uppercase tracking-widest"
                  >
                    Engineered Features
                  </Text>
                  <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {acceptedEngineeredFeatures.map((feature) => (
                      <div
                        key={feature.name}
                        className="flex flex-col gap-1 border-l-2 border-kumo-success/40 pl-3"
                      >
                        <Text size="xs" bold DANGEROUS_className="truncate">
                          {feature.name}
                        </Text>
                        <Text size="xs" variant="secondary">
                          {feature.expectedBenefit}
                        </Text>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="mt-8 flex flex-col gap-4 border-t border-kumo-line pt-6 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-kumo-success animate-pulse" />
                  <Text size="sm" bold>
                    Pipeline Verified
                  </Text>
                </div>
                <Text
                  size="xs"
                  variant="secondary"
                  DANGEROUS_className="max-w-md"
                >
                  Feature engineering steps are mapped. The target column
                  remains in its original form to prevent leakage during
                  training.
                </Text>
                {transformState.status === "error" && (
                  <Text size="xs" DANGEROUS_className="text-kumo-danger mt-1">
                    {transformState.message}
                  </Text>
                )}
                {transformState.status === "ready" && (
                  <Text
                    size="xs"
                    DANGEROUS_className="text-kumo-success mt-1 font-medium"
                  >
                    Successfully exported{" "}
                    {transformState.rowCount.toLocaleString()} rows with{" "}
                    {transformState.columns.toLocaleString()} columns.
                  </Text>
                )}
              </div>
              <Button
                type="button"
                variant="primary"
                size="base"
                className="px-6"
                icon={<DownloadSimpleIcon size={16} />}
                disabled={
                  !targetColumn ||
                  !isSelectionFinalized ||
                  transformState.status === "running"
                }
                onClick={onDownloadTransformed}
              >
                {transformState.status === "running"
                  ? "Processing..."
                  : "Download Processed CSV"}
              </Button>
            </div>
            {transformState.status === "ready" && (
              <div className="mt-6 rounded-xl border border-kumo-line bg-kumo-base/50 p-5">
                <div className="flex items-center gap-2 mb-3">
                  <GearIcon size={14} className="text-kumo-inactive" />
                  <Text
                    size="xs"
                    bold
                    variant="secondary"
                    DANGEROUS_className="uppercase tracking-widest"
                  >
                    Post-Process Audit
                  </Text>
                </div>
                <div className="grid gap-2">
                  {transformState.audit.map((item) => (
                    <div
                      key={item}
                      className="flex items-start gap-2 text-xs text-kumo-subtle"
                    >
                      <CheckCircleIcon
                        size={12}
                        className="mt-0.5 text-kumo-success"
                      />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SplitConfigPanel({
  fileName,
  rowCount,
  onConfirm,
  onCancel
}: {
  fileName: string;
  rowCount: number;
  onConfirm: (config: SplitConfig) => void;
  onCancel: () => void;
}) {
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [seed, setSeed] = useState(42);
  const { trainCount, testCount } = countSplitRows(rowCount, seed, trainRatio);

  return (
    <div className="rounded-lg border border-kumo-line bg-kumo-elevated p-4">
      <div className="mb-2">
        <Text size="sm" bold>
          Train/test split
        </Text>
      </div>
      <div className="mb-3">
        <Text size="xs" variant="secondary">
          {fileName} &mdash; {rowCount.toLocaleString()} rows total. Choose a
          split before profiling. Only training rows will be analysed and
          previewed.
        </Text>
      </div>

      <div className="mb-3">
        <Text size="xs" variant="secondary">
          Train ratio: {Math.round(trainRatio * 100)}%
        </Text>
        <input
          type="range"
          min={0.5}
          max={0.9}
          step={0.05}
          value={trainRatio}
          className="w-full accent-kumo-brand"
          onChange={(e) => setTrainRatio(Number(e.target.value))}
        />
        <div className="flex justify-between text-xs text-kumo-subtle">
          <span>{trainCount.toLocaleString()} train</span>
          <span>{testCount.toLocaleString()} held out</span>
        </div>
      </div>

      <div className="mb-4">
        <Text size="xs" variant="secondary">
          Seed
        </Text>
        <input
          type="number"
          value={seed}
          min={0}
          className="mt-1 h-8 w-24 rounded-md border border-kumo-line bg-kumo-base px-2 text-sm text-kumo-default outline-none focus:ring-2 focus:ring-kumo-ring"
          onChange={(e) => setSeed(Number(e.target.value) || 0)}
        />
      </div>

      <div className="flex gap-2">
        <Button
          type="button"
          variant="primary"
          size="sm"
          onClick={() => onConfirm({ trainRatio, seed })}
        >
          Confirm split &amp; profile
        </Button>
        <Button type="button" variant="secondary" size="sm" onClick={onCancel}>
          Cancel
        </Button>
      </div>
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
  const [columnFilter, setColumnFilter] = useState<ColumnFilter>("all");
  const [columnSort, setColumnSort] = useState<ColumnSort>({
    key: "notesCount",
    direction: "desc"
  });
  const [aiReviewState, setAiReviewState] = useState<AiReviewState>({
    status: "idle"
  });
  const [preprocessingReviewState, setPreprocessingReviewState] =
    useState<PreprocessingReviewState>({
      status: "idle"
    });
  const [featureSuggestionState, setFeatureSuggestionState] =
    useState<FeatureSuggestionState>({ status: "idle" });
  const [transformState, setTransformState] = useState<TransformState>({
    status: "idle"
  });
  const [targetColumn, setTargetColumn] = useState<string | null>(null);
  const [columnActions, setColumnActions] = useState<
    Record<string, ColumnPreparationAction>
  >({});
  const [finalizedFeatureColumns, setFinalizedFeatureColumns] = useState<
    string[] | null
  >(null);
  const [preprocessingChoices, setPreprocessingChoices] =
    useState<PreprocessingChoices>({});

  const currentSummary =
    uploadState.status === "ready" ? uploadState.summary : null;
  const currentFile = uploadState.status === "ready" ? uploadState.file : null;
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
  const updateColumnSort = useCallback((key: ColumnSortKey) => {
    setColumnSort((current) => ({
      key,
      direction:
        current.key === key && current.direction === "desc" ? "asc" : "desc"
    }));
  }, []);

  useEffect(() => {
    const context = currentSummary
      ? buildActiveDatasetContext({
          summary: currentSummary,
          targetColumn,
          columnActions,
          finalizedFeatureColumns,
          preprocessingChoices,
          preprocessingPlan:
            preprocessingReviewState.status === "ready"
              ? preprocessingReviewState.plan
              : null
        })
      : null;

    window.dispatchEvent(
      new CustomEvent("automated-fe:dataset-context", {
        detail: { context }
      })
    );
  }, [
    columnActions,
    currentSummary,
    finalizedFeatureColumns,
    preprocessingChoices,
    preprocessingReviewState,
    targetColumn
  ]);

  const resetWorkspace = useCallback(() => {
    setUploadState({ status: "idle" });
    setAiReviewState({ status: "idle" });
    setPreprocessingReviewState({ status: "idle" });
    setFeatureSuggestionState({ status: "idle" });
    setTransformState({ status: "idle" });
    setTargetColumn(null);
    setColumnActions({});
    setFinalizedFeatureColumns(null);
    setPreprocessingChoices({});
    setColumnFilter("all");
    setColumnSort({ key: "notesCount", direction: "desc" });
    if (csvInputRef.current) csvInputRef.current.value = "";
  }, []);

  const generateAiReview = useCallback(async () => {
    if (!currentSummary) return;

    setAiReviewState({ status: "loading" });
    setPreprocessingReviewState({ status: "idle" });
    setFeatureSuggestionState({ status: "idle" });
    setTransformState({ status: "idle" });
    setFinalizedFeatureColumns(null);
    try {
      const response = await fetch("/api/column-selection", {
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
            "The column selection review could not be generated."
        );
      }

      const reviewPlan = data as ColumnSelectionPlan;
      const suggestedTarget =
        getSuggestedTarget(reviewPlan)?.columnName ?? null;
      const nextColumnActions = Object.fromEntries(
        currentSummary.columns.map((column) => {
          const decision = getDecisionForColumn(reviewPlan, column.name);
          return [
            column.name,
            getColumnPreparationAction(column, decision, suggestedTarget)
          ];
        })
      );

      setAiReviewState({
        status: "ready",
        plan: reviewPlan
      });
      setTargetColumn(suggestedTarget);
      setColumnActions(nextColumnActions);
      setFinalizedFeatureColumns(null);
      setPreprocessingChoices({});
      toasts.add({
        title: "Column suggestions generated",
        description: "Confirm the target and columns to keep."
      });
    } catch (error) {
      setAiReviewState({
        status: "error",
        message:
          error instanceof Error
            ? error.message
            : "The column selection review could not be generated."
      });
    }
  }, [currentSummary, toasts]);

  const sendAiReviewToChat = useCallback(() => {
    if (!currentSummary || aiReviewState.status !== "ready") return;

    window.dispatchEvent(
      new CustomEvent("automated-fe:agent-chat-message", {
        detail: {
          text: buildReviewChatMessage(
            currentSummary,
            aiReviewState.plan,
            preprocessingReviewState.status === "ready"
              ? preprocessingReviewState.plan
              : null
          )
        }
      })
    );
    toasts.add({
      title: "Review sent to chat",
      description: "The agent sidebar now has the current review context."
    });
  }, [aiReviewState, currentSummary, preprocessingReviewState, toasts]);

  const acceptTarget = useCallback((columnName: string) => {
    setPreprocessingReviewState({ status: "idle" });
    setFeatureSuggestionState({ status: "idle" });
    setTransformState({ status: "idle" });
    setFinalizedFeatureColumns(null);
    setPreprocessingChoices({});
    setTargetColumn(columnName);
    setColumnActions((current) => ({
      ...current,
      [columnName]: "feature"
    }));
    setPreprocessingChoices((current) => ({
      ...current,
      [columnName]: []
    }));
  }, []);

  const changeTarget = useCallback((columnName: string) => {
    setPreprocessingReviewState({ status: "idle" });
    setFeatureSuggestionState({ status: "idle" });
    setTransformState({ status: "idle" });
    setFinalizedFeatureColumns(null);
    setPreprocessingChoices({});
    setTargetColumn(columnName || null);
    if (!columnName) return;
    setColumnActions((current) => ({
      ...current,
      [columnName]: "feature"
    }));
    setPreprocessingChoices((current) => ({
      ...current,
      [columnName]: []
    }));
  }, []);

  const changeColumnAction = useCallback(
    (columnName: string, action: ColumnPreparationAction) => {
      setColumnActions((current) => ({
        ...current,
        [columnName]: action
      }));
      setPreprocessingReviewState({ status: "idle" });
      setFeatureSuggestionState({ status: "idle" });
      setTransformState({ status: "idle" });
      setFinalizedFeatureColumns(null);
      if (action === "drop") {
        setPreprocessingChoices((current) => ({
          ...current,
          [columnName]: []
        }));
      }
    },
    []
  );

  const finishColumnSelection = useCallback(() => {
    if (!currentSummary || !targetColumn) return;

    const keptColumns = currentSummary.columns
      .map((column) => column.name)
      .filter(
        (columnName) =>
          columnName !== targetColumn && columnActions[columnName] === "feature"
      );

    setFinalizedFeatureColumns(keptColumns);
    setPreprocessingReviewState({ status: "idle" });
    setFeatureSuggestionState({ status: "idle" });
    setTransformState({ status: "idle" });
    setPreprocessingChoices({});
    toasts.add({
      title: "Column selection finished",
      description: `${keptColumns.length.toLocaleString()} feature column${
        keptColumns.length === 1 ? "" : "s"
      } kept for preprocessing.`
    });
  }, [columnActions, currentSummary, targetColumn, toasts]);

  const changePreprocessingChoice = useCallback(
    (columnName: string, choice: PreprocessingChoice, checked: boolean) => {
      setTransformState({ status: "idle" });
      setPreprocessingChoices((current) => {
        const existing = current[columnName] ?? [];
        const nextChoices = checked
          ? Array.from(new Set([...existing, choice]))
          : existing.filter((item) => item !== choice);

        return {
          ...current,
          [columnName]: nextChoices
        };
      });
      if (isDropPreprocessingChoice(choice)) {
        setColumnActions((current) => ({
          ...current,
          [columnName]: "drop"
        }));
      }
    },
    []
  );

  const downloadTransformedDataset = useCallback(async () => {
    if (!currentFile || !currentSummary || !targetColumn) return;

    const featureColumns =
      finalizedFeatureColumns ??
      currentSummary.columns
        .map((column) => column.name)
        .filter(
          (columnName) =>
            columnName !== targetColumn &&
            columnActions[columnName] === "feature"
        );
    const preprocessingSteps = buildSelectedPreprocessingSteps(
      preprocessingChoices
    ).filter((step) => featureColumns.includes(step.columnName));

    setTransformState({ status: "running" });
    try {
      const splitConfig: SplitConfig | undefined =
        uploadState.status === "ready" ? uploadState.splitConfig : undefined;
      const result = await transformCsvFile(
        currentFile,
        {
          targetColumn,
          featureColumns,
          preprocessingSteps,
          engineeredFeatures: getAcceptedFeatureSuggestions(
            featureSuggestionState
          )
        },
        splitConfig
      );

      if ("trainCsv" in result) {
        // Split export — download train CSV
        const trainBlob = new Blob([result.trainCsv], {
          type: "text/csv;charset=utf-8"
        });
        const trainUrl = URL.createObjectURL(trainBlob);
        const trainLink = document.createElement("a");
        const baseName = currentSummary.fileName.replace(/\.csv$/i, "");

        trainLink.href = trainUrl;
        trainLink.download = `${baseName || "dataset"}-train.csv`;
        document.body.appendChild(trainLink);
        trainLink.click();
        trainLink.remove();
        URL.revokeObjectURL(trainUrl);

        // Download test CSV
        const testBlob = new Blob([result.testCsv], {
          type: "text/csv;charset=utf-8"
        });
        const testUrl = URL.createObjectURL(testBlob);
        const testLink = document.createElement("a");
        testLink.href = testUrl;
        testLink.download = `${baseName || "dataset"}-test.csv`;
        document.body.appendChild(testLink);
        testLink.click();
        testLink.remove();
        URL.revokeObjectURL(testUrl);

        setTransformState({
          status: "ready",
          audit: result.audit,
          rowCount: result.trainRowCount,
          columns: result.outputColumns.length
        });
        toasts.add({
          title: "Train/test CSVs exported",
          description: `${result.trainRowCount.toLocaleString()} train + ${result.testRowCount.toLocaleString()} test rows.`
        });
      } else {
        // Single export (no split)
        const blob = new Blob([result.csv], {
          type: "text/csv;charset=utf-8"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        const baseName = currentSummary.fileName.replace(/\.csv$/i, "");

        link.href = url;
        link.download = `${baseName || "dataset"}-transformed.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setTransformState({
          status: "ready",
          audit: result.audit,
          rowCount: result.rowCount,
          columns: result.outputColumns.length
        });
        toasts.add({
          title: "Transformed CSV ready",
          description: `${result.rowCount.toLocaleString()} rows exported.`
        });
      }
    } catch (error) {
      setTransformState({
        status: "error",
        message:
          error instanceof Error
            ? error.message
            : "The transformed CSV could not be generated."
      });
    }
  }, [
    columnActions,
    currentFile,
    currentSummary,
    finalizedFeatureColumns,
    preprocessingChoices,
    featureSuggestionState,
    targetColumn,
    toasts,
    uploadState
  ]);

  const handleSplitConfirm = useCallback(
    async (splitConfig: SplitConfig) => {
      const currentFile =
        uploadState.status === "choosing_split" ? uploadState.file : null;
      if (!currentFile) return;

      setUploadState({ status: "parsing", fileName: currentFile.name });
      try {
        const summary = await parseCsvFile(currentFile, splitConfig);
        setUploadState({
          status: "ready",
          summary,
          file: currentFile,
          splitConfig,
          rowCount:
            uploadState.status === "choosing_split" ? uploadState.rowCount : 0
        });
        setAiReviewState({ status: "idle" });
        setPreprocessingReviewState({ status: "idle" });
        setFeatureSuggestionState({ status: "idle" });
        setTransformState({ status: "idle" });
        setTargetColumn(null);
        setColumnActions({});
        setFinalizedFeatureColumns(null);
        setPreprocessingChoices({});
        const heldOut =
          uploadState.status === "choosing_split"
            ? uploadState.rowCount - summary.parsedRowCount
            : 0;
        toasts.add({
          title: "Training set profiled",
          description: `${summary.parsedRowCount.toLocaleString()} training rows (${heldOut.toLocaleString()} held out as test).`
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
    [toasts, uploadState]
  );

  const handleSplitCancel = useCallback(() => {
    setUploadState({ status: "idle" });
    if (csvInputRef.current) csvInputRef.current.value = "";
  }, []);

  const generatePreprocessingReview = useCallback(async () => {
    if (!currentSummary || !targetColumn) return;

    const keptColumns =
      finalizedFeatureColumns ??
      currentSummary.columns
        .map((column) => column.name)
        .filter(
          (columnName) =>
            columnName !== targetColumn &&
            columnActions[columnName] === "feature"
        );

    setPreprocessingReviewState({ status: "loading" });
    setTransformState({ status: "idle" });
    try {
      const response = await fetch("/api/preprocessing-review", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          profile: buildAiReviewProfile(currentSummary),
          targetColumn,
          keptColumns
        })
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          (data as { error?: string } | null)?.error ||
            "The preprocessing review could not be generated."
        );
      }

      const plan = data as ColumnPreprocessingPlan;
      setPreprocessingReviewState({ status: "ready", plan });
      setPreprocessingChoices(
        Object.fromEntries(
          keptColumns.map((columnName) => {
            const suggestion = plan.preprocessingSuggestions.find(
              (item) => item.columnName === columnName
            );
            return [
              columnName,
              suggestion && !isNoPreprocessingChoice(suggestion.action)
                ? [suggestion.action]
                : []
            ];
          })
        )
      );
      toasts.add({
        title: "Preprocessing suggestions generated",
        description: "Review the proposed transformations for kept columns."
      });
    } catch (error) {
      setPreprocessingReviewState({
        status: "error",
        message:
          error instanceof Error
            ? error.message
            : "The preprocessing review could not be generated."
      });
    }
  }, [
    columnActions,
    currentSummary,
    finalizedFeatureColumns,
    targetColumn,
    toasts
  ]);

  const generateFeatureSuggestions = useCallback(async () => {
    if (
      !currentSummary ||
      !targetColumn ||
      !finalizedFeatureColumns ||
      preprocessingReviewState.status !== "ready"
    ) {
      return;
    }

    const keptColumns = finalizedFeatureColumns.filter(
      (columnName) => columnName !== targetColumn
    );
    const preprocessingSteps = buildSelectedPreprocessingSteps(
      preprocessingChoices
    ).filter((step) => keptColumns.includes(step.columnName));

    setFeatureSuggestionState({ status: "loading" });
    try {
      const response = await fetch("/api/feature-suggestions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          profile: buildAiReviewProfile(currentSummary),
          targetColumn,
          keptColumns,
          preprocessingSteps
        })
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          (data as { error?: string } | null)?.error ||
            "The feature suggestions could not be generated."
        );
      }

      const result = data as FeatureValidationResult;
      setFeatureSuggestionState({ status: "ready", result, decisions: {} });
      toasts.add({
        title: "Feature suggestions generated",
        description: "Review suggested features."
      });
    } catch (error) {
      setFeatureSuggestionState({
        status: "error",
        message: error instanceof Error ? error.message : String(error)
      });
    }
  }, [
    currentSummary,
    finalizedFeatureColumns,
    preprocessingChoices,
    preprocessingReviewState.status,
    targetColumn,
    toasts
  ]);

  const decideFeatureSuggestion = useCallback(
    (featureKey: string, decision: FeatureSuggestionDecision) => {
      setTransformState({ status: "idle" });
      setFeatureSuggestionState((current) => {
        if (current.status !== "ready") return current;

        return {
          ...current,
          decisions: {
            ...current.decisions,
            [featureKey]: decision
          }
        };
      });
    },
    []
  );

  const handleCsvFile = useCallback(
    async (file: File) => {
      if (uploadState.status === "ready") {
        const replace = window.confirm(
          "Uploading a new CSV will replace the active dataset and preprocessing pipeline."
        );
        if (!replace) return;
        setAiReviewState({ status: "idle" });
        setPreprocessingReviewState({ status: "idle" });
        setFeatureSuggestionState({ status: "idle" });
        setTransformState({ status: "idle" });
        setTargetColumn(null);
        setColumnActions({});
        setFinalizedFeatureColumns(null);
        setPreprocessingChoices({});
      }

      setUploadState({ status: "validating", fileName: file.name });
      const validationError = validateCsvFile(file);
      if (validationError) {
        setUploadState({ status: "error", message: validationError });
        return;
      }

      // Quick row count via file text to show split config
      try {
        const text = await file.text();
        const rowCount =
          text.split("\n").filter((line) => line.trim()).length - 1;
        setUploadState({
          status: "choosing_split",
          fileName: file.name,
          file,
          rowCount: Math.max(0, rowCount)
        });
      } catch {
        setUploadState({
          status: "error",
          message: "Could not read the CSV file."
        });
      }
    },
    [uploadState.status]
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

        <div
          className={`relative group rounded-xl border-2 border-dashed p-8 transition-all duration-200 text-center ${
            isCsvDragging
              ? "border-kumo-brand bg-kumo-brand/5"
              : "border-kumo-line bg-kumo-elevated hover:border-kumo-brand/30"
          }`}
        >
          <div className="flex flex-col items-center gap-3">
            <div
              className={`p-4 rounded-full transition-colors ${isCsvDragging ? "bg-kumo-brand/20 text-kumo-brand" : "bg-kumo-base text-kumo-inactive group-hover:text-kumo-brand/70"}`}
            >
              <DownloadSimpleIcon size={32} />
            </div>
            <div>
              <Text size="base" bold>
                {uploadState.status === "idle"
                  ? "Drop your CSV here"
                  : "Change dataset"}
              </Text>
              <Text
                size="sm"
                variant="secondary"
                DANGEROUS_className="mt-1 max-w-md"
              >
                Drag and drop a file, or click to browse. Supports `.csv` up to{" "}
                {formatBytes(MAX_CSV_SIZE_BYTES)}.
              </Text>
            </div>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              icon={<PaperclipIcon size={14} />}
              onClick={() => csvInputRef.current?.click()}
              disabled={
                uploadState.status === "validating" ||
                uploadState.status === "parsing"
              }
            >
              Choose File
            </Button>
          </div>
        </div>

        {uploadState.status === "validating" && (
          <Badge variant="secondary" className="animate-pulse">
            Validating {uploadState.fileName}...
          </Badge>
        )}
        {uploadState.status === "choosing_split" && (
          <SplitConfigPanel
            fileName={uploadState.fileName}
            rowCount={uploadState.rowCount}
            onConfirm={handleSplitConfirm}
            onCancel={handleSplitCancel}
          />
        )}
        {uploadState.status === "parsing" && (
          <Badge variant="secondary" className="animate-pulse">
            Parsing {uploadState.fileName}...
          </Badge>
        )}
        {uploadState.status === "error" && (
          <div className="rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 px-3 py-2">
            <Text size="sm">{uploadState.message}</Text>
          </div>
        )}

        {currentSummary ? (
          <div className="grid gap-6">
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="flex flex-col gap-1 rounded-xl border border-kumo-line bg-kumo-elevated p-4 transition-all hover:border-kumo-brand/50">
                <Text
                  size="xs"
                  variant="secondary"
                  bold
                  DANGEROUS_className="uppercase tracking-wider"
                >
                  File Name
                </Text>
                <Text size="base" bold DANGEROUS_className="truncate">
                  {currentSummary.fileName}
                </Text>
                <Text size="xs" variant="secondary">
                  {formatBytes(currentSummary.fileSizeBytes)}
                </Text>
              </div>
              <div className="flex flex-col gap-1 rounded-xl border border-kumo-line bg-kumo-elevated p-4 transition-all hover:border-kumo-brand/50">
                <Text
                  size="xs"
                  variant="secondary"
                  bold
                  DANGEROUS_className="uppercase tracking-wider"
                >
                  Training Rows
                </Text>
                <div className="flex items-baseline gap-2">
                  <Text size="lg" bold>
                    {currentSummary.parsedRowCount.toLocaleString()}
                  </Text>
                  <Text size="xs" variant="secondary">
                    rows
                  </Text>
                </div>
                {uploadState.status === "ready" && uploadState.splitConfig && (
                  <Text size="xs" DANGEROUS_className="text-kumo-success">
                    {uploadState.rowCount.toLocaleString()} total rows split
                    into {currentSummary.parsedRowCount.toLocaleString()} train
                    and{" "}
                    {(
                      uploadState.rowCount - currentSummary.parsedRowCount
                    ).toLocaleString()}{" "}
                    test rows.
                  </Text>
                )}
              </div>
              <div className="flex flex-col gap-1 rounded-xl border border-kumo-line bg-kumo-elevated p-4 transition-all hover:border-kumo-brand/50">
                <Text
                  size="xs"
                  variant="secondary"
                  bold
                  DANGEROUS_className="uppercase tracking-wider"
                >
                  Held Out (Test)
                </Text>
                <div className="flex items-baseline gap-2">
                  <Text size="lg" bold>
                    {uploadState.status === "ready" && uploadState.splitConfig
                      ? (
                          uploadState.rowCount - currentSummary.parsedRowCount
                        ).toLocaleString()
                      : "0"}
                  </Text>
                  <Text size="xs" variant="secondary">
                    rows
                  </Text>
                </div>
                <Text size="xs" variant="secondary">
                  Excluded from profiling
                </Text>
              </div>
              <div className="flex flex-col gap-1 rounded-xl border border-kumo-line bg-kumo-elevated p-4 transition-all hover:border-kumo-brand/50">
                <Text
                  size="xs"
                  variant="secondary"
                  bold
                  DANGEROUS_className="uppercase tracking-wider"
                >
                  Total Columns
                </Text>
                <div className="flex items-baseline gap-2">
                  <Text size="lg" bold>
                    {currentSummary.columns.length}
                  </Text>
                  <Text size="xs" variant="secondary">
                    features
                  </Text>
                </div>
                <Text size="xs" variant="secondary">
                  {columnQualityCounts.hasNotes} with warnings
                </Text>
              </div>
            </div>

            {(currentSummary.warnings.length > 0 ||
              (uploadState.status === "ready" && uploadState.splitConfig)) && (
              <div className="rounded-lg border border-kumo-warning/40 bg-kumo-warning/10 px-3 py-2">
                {uploadState.status === "ready" && uploadState.splitConfig && (
                  <Text size="xs">
                    Test set (
                    {(
                      uploadState.rowCount - currentSummary.parsedRowCount
                    ).toLocaleString()}{" "}
                    rows) is held out and excluded from all profiling, preview,
                    and statistics.
                  </Text>
                )}
                {currentSummary.warnings.map((warning) => (
                  <Text key={warning} size="xs">
                    {warning}
                  </Text>
                ))}
              </div>
            )}
            <CollapsibleSection
              title="Column Quality Analysis"
              subtitle="Inspect and filter columns based on profiling results."
              defaultOpen={true}
            >
              <div className="flex flex-col gap-4 border-b border-kumo-line bg-kumo-base/30 p-5">
                <div className="flex flex-wrap gap-2">
                  {COLUMN_FILTERS.map((filter) => (
                    <button
                      key={filter.key}
                      type="button"
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        columnFilter === filter.key
                          ? "bg-kumo-brand text-kumo-inverse shadow-sm"
                          : "bg-kumo-base text-kumo-subtle hover:bg-kumo-line/50"
                      }`}
                      onClick={() => setColumnFilter(filter.key)}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto scrollbar-thin scrollbar-thumb-kumo-line scrollbar-track-transparent">
                <table className="min-w-full text-left text-sm">
                  <thead className="bg-kumo-base/50 text-kumo-subtle backdrop-blur-sm">
                    <tr>
                      <SortableColumnHeader
                        label="Column Name"
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
                        label="% Missing"
                        sortKey="missingPercent"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <SortableColumnHeader
                        label="Populated"
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
                        label="Cardinality"
                        sortKey="uniqueRatio"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Top Values
                      </th>
                      <th className="px-4 py-3 font-semibold uppercase tracking-wider text-[10px]">
                        Samples
                      </th>
                      <SortableColumnHeader
                        label="Notes"
                        sortKey="notesCount"
                        currentSort={columnSort}
                        onSort={updateColumnSort}
                      />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-kumo-line/50">
                    {visibleColumns.map((column) => (
                      <tr
                        key={column.name}
                        className="hover:bg-kumo-base/20 transition-colors"
                      >
                        <td className="px-4 py-4 font-bold text-kumo-default">
                          {column.name}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.inferredType}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.missingCount.toLocaleString()}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {formatPercent(
                            getMissingPercent(
                              column,
                              currentSummary.parsedRowCount
                            )
                          )}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.nonMissingCount.toLocaleString()}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.uniqueCount.toLocaleString()}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {formatPercent(column.uniqueRatio * 100)}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.topValues.length > 0 ? (
                            <div className="flex flex-col gap-1 max-w-[200px]">
                              {column.topValues.map((topValue) => (
                                <span
                                  key={topValue.value}
                                  className="truncate"
                                  title={`${topValue.value} (${topValue.count})`}
                                >
                                  <span className="font-medium text-kumo-default">
                                    {topValue.value}
                                  </span>
                                  <span className="ml-1 opacity-60">
                                    ({formatPercent(topValue.percent)})
                                  </span>
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="italic opacity-40">Empty</span>
                          )}
                        </td>
                        <td className="px-4 py-4 text-xs text-kumo-subtle">
                          {column.sampleValues.length > 0 ? (
                            <div
                              className="truncate max-w-[200px]"
                              title={column.sampleValues.join(", ")}
                            >
                              {column.sampleValues.join(", ")}
                            </div>
                          ) : (
                            <span className="italic opacity-40">None</span>
                          )}
                        </td>
                        <td className="px-4 py-4">
                          {column.profilingNotes.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {column.profilingNotes.map((note) => (
                                <Badge
                                  key={note.code}
                                  variant="secondary"
                                  className="text-[9px] py-0 px-1 font-medium bg-kumo-warning/10 text-kumo-warning border-kumo-warning/20"
                                >
                                  {note.code.replaceAll("_", " ")}
                                </Badge>
                              ))}
                            </div>
                          ) : (
                            <CheckCircleIcon
                              size={14}
                              className="text-kumo-success opacity-40"
                            />
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
            </CollapsibleSection>

            <CollapsibleSection
              title="Data Preview"
              subtitle="Inspect the first few rows of the dataset."
              defaultOpen={false}
            >
              <PreviewTable
                columns={currentSummary.columns}
                rows={currentSummary.previewRows}
              />
            </CollapsibleSection>

            <PreparationReviewPanel
              summary={currentSummary}
              reviewState={aiReviewState}
              preprocessingState={preprocessingReviewState}
              featureSuggestionState={featureSuggestionState}
              onGenerateFeatures={() => void generateFeatureSuggestions()}
              onFeatureSuggestionDecision={decideFeatureSuggestion}
              targetColumn={targetColumn}
              columnActions={columnActions}
              finalizedFeatureColumns={finalizedFeatureColumns}
              preprocessingChoices={preprocessingChoices}
              transformState={transformState}
              onGenerate={() => void generateAiReview()}
              onGeneratePreprocessing={() => void generatePreprocessingReview()}
              onDownloadTransformed={() => void downloadTransformedDataset()}
              onAcceptTarget={acceptTarget}
              onTargetChange={changeTarget}
              onFinishSelection={finishColumnSelection}
              onColumnActionChange={changeColumnAction}
              onPreprocessingChoiceChange={changePreprocessingChoice}
              onSendToChat={sendAiReviewToChat}
            />
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
  const [showChat, setShowChat] = useState(false);

  return (
    <div className="min-h-screen bg-kumo-base text-kumo-default">
      <header className="sticky top-0 z-20 border-b border-kumo-line bg-kumo-base/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex flex-col">
            <Text size="lg" bold DANGEROUS_className="leading-tight">
              Automated FE
            </Text>
            <Text size="xs" variant="secondary">
              CSV profiling and preprocessing workspace
            </Text>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              className="lg:hidden"
              icon={<ChatCircleDotsIcon size={18} />}
              onClick={() => setShowChat(!showChat)}
            >
              Chat
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-5 py-6">
        <div className="min-w-0 space-y-6">
          <DatasetWorkspace />
        </div>
        <aside
          className={`fixed inset-y-0 right-0 z-40 w-full max-w-[400px] border-l border-kumo-line bg-kumo-base p-5 shadow-2xl transition-transform duration-300 ease-in-out lg:w-[380px] lg:max-w-none ${
            showChat ? "translate-x-0" : "translate-x-full"
          }`}
          aria-hidden={!showChat}
        >
          <div className="flex h-full flex-col">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ChatCircleDotsIcon size={20} />
                <Text bold>Agent Chat</Text>
              </div>
              <Button
                variant="ghost"
                shape="square"
                size="sm"
                icon={<XIcon size={18} />}
                aria-label="Close chat"
                onClick={() => setShowChat(false)}
              />
            </div>
            <AgentChatSidebar />
          </div>
        </aside>

        {/* Floating toggle button (visible when chat is collapsed) */}
        <div className="fixed bottom-5 right-5 z-50 lg:bottom-8 lg:right-8">
          {!showChat && (
            <Button
              variant="primary"
              shape="square"
              size="lg"
              icon={<ChatCircleDotsIcon size={20} />}
              aria-label="Open chat"
              onClick={() => setShowChat(true)}
            />
          )}
        </div>
      </main>

      {showChat && (
        <button
          type="button"
          aria-label="Close chat overlay"
          className="fixed inset-0 z-30 w-full h-full bg-kumo-contrast/10 backdrop-blur-[2px] cursor-default border-none p-0 outline-none"
          onClick={() => setShowChat(false)}
        />
      )}
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
