import { Suspense, useCallback, useState, useEffect, useRef } from "react";
import { useAgent } from "agents/react";
import { useAgentChat } from "@cloudflare/ai-chat/react";
import { getToolName, isToolUIPart, type UIMessage } from "ai";
import type { MCPServersState } from "agents";
import type { ChatAgent } from "./server";
import {
  Badge,
  Button,
  Empty,
  InputArea,
  Surface,
  Switch,
  Text
} from "@cloudflare/kumo";
import { Toasty, useKumoToastManager } from "@cloudflare/kumo/components/toast";
import { Streamdown } from "streamdown";
import { code } from "@streamdown/code";
import {
  PaperPlaneRightIcon,
  StopIcon,
  TrashIcon,
  GearIcon,
  ChatCircleDotsIcon,
  CircleIcon,
  MoonIcon,
  SunIcon,
  CheckCircleIcon,
  XCircleIcon,
  BrainIcon,
  CaretDownIcon,
  BugIcon,
  PlugsConnectedIcon,
  PlusIcon,
  SignInIcon,
  XIcon,
  WrenchIcon,
  PaperclipIcon,
  ImageIcon
} from "@phosphor-icons/react";

const MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024;

type InferredColumnType =
  | "string"
  | "number"
  | "boolean"
  | "date"
  | "empty"
  | "mixed";

type PreviewValue = string | number | boolean | null;

interface ColumnSummary {
  name: string;
  inferredType: InferredColumnType;
  missingCount: number;
  sampleValues: string[];
}

interface DatasetSummary {
  fileName: string;
  fileSizeBytes: number;
  parsedRowCount: number;
  columns: ColumnSummary[];
  previewRows: Record<string, PreviewValue>[];
  warnings: string[];
}

type UploadState =
  | { status: "idle" }
  | { status: "validating"; fileName: string }
  | { status: "parsing"; fileName: string }
  | { status: "ready"; summary: DatasetSummary }
  | { status: "error"; message: string };

// ── Attachment helpers ────────────────────────────────────────────────

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

// ── CSV upload helpers ────────────────────────────────────────────────

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function validateCsvFile(file: File): string | null {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Select a .csv file.";
  }
  if (file.size === 0) {
    return "The CSV file is empty.";
  }
  if (file.size > MAX_CSV_SIZE_BYTES) {
    return `The CSV file is ${formatBytes(file.size)}. The MVP limit is ${formatBytes(MAX_CSV_SIZE_BYTES)}.`;
  }
  return null;
}

async function parseCsvFile(file: File): Promise<DatasetSummary> {
  const response = await fetch("/profile-csv", {
    method: "POST",
    headers: {
      "content-type": file.type || "text/csv",
      "x-file-name": encodeURIComponent(file.name),
      "x-file-size": String(file.size)
    },
    body: file
  });

  const body = (await response.json()) as DatasetSummary | { error?: string };
  if (!response.ok) {
    throw new Error(
      "error" in body && body.error ? body.error : "CSV parsing failed."
    );
  }

  const summary = body as DatasetSummary;
  console.info("CSV upload parsed", {
    fileName: summary.fileName,
    fileSizeBytes: summary.fileSizeBytes,
    parsedRowCount: summary.parsedRowCount,
    columns: summary.columns.length
  });

  return summary;
}

// ── Small components ──────────────────────────────────────────────────

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

// ── Tool rendering ────────────────────────────────────────────────────

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

  // Completed
  if (part.state === "output-available") {
    return (
      <div className="flex justify-start">
        <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
          <div className="flex items-center gap-2 mb-1">
            <GearIcon size={14} className="text-kumo-inactive" />
            <Text size="xs" variant="secondary" bold>
              {toolName}
            </Text>
            <Badge variant="secondary">Done</Badge>
          </div>
          <div className="font-mono">
            <Text size="xs" variant="secondary">
              {JSON.stringify(part.output, null, 2)}
            </Text>
          </div>
        </Surface>
      </div>
    );
  }

  // Needs approval
  if ("approval" in part && part.state === "approval-requested") {
    const approvalId = (part.approval as { id?: string })?.id;
    return (
      <div className="flex justify-start">
        <Surface className="max-w-[85%] px-4 py-3 rounded-xl ring-2 ring-kumo-warning">
          <div className="flex items-center gap-2 mb-2">
            <GearIcon size={14} className="text-kumo-warning" />
            <Text size="sm" bold>
              Approval needed: {toolName}
            </Text>
          </div>
          <div className="font-mono mb-3">
            <Text size="xs" variant="secondary">
              {JSON.stringify(part.input, null, 2)}
            </Text>
          </div>
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
        </Surface>
      </div>
    );
  }

  // Rejected / denied
  if (
    part.state === "output-denied" ||
    ("approval" in part &&
      (part.approval as { approved?: boolean })?.approved === false)
  ) {
    return (
      <div className="flex justify-start">
        <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
          <div className="flex items-center gap-2">
            <XCircleIcon size={14} className="text-kumo-danger" />
            <Text size="xs" variant="secondary" bold>
              {toolName}
            </Text>
            <Badge variant="secondary">Rejected</Badge>
          </div>
        </Surface>
      </div>
    );
  }

  // Executing
  if (part.state === "input-available" || part.state === "input-streaming") {
    return (
      <div className="flex justify-start">
        <Surface className="max-w-[85%] px-4 py-2.5 rounded-xl ring ring-kumo-line">
          <div className="flex items-center gap-2">
            <GearIcon size={14} className="text-kumo-inactive animate-spin" />
            <Text size="xs" variant="secondary">
              Running {toolName}...
            </Text>
          </div>
        </Surface>
      </div>
    );
  }

  return null;
}

// ── Main chat ─────────────────────────────────────────────────────────

function Chat() {
  const [connected, setConnected] = useState(false);
  const [input, setInput] = useState("");
  const [showDebug, setShowDebug] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isCsvDragging, setIsCsvDragging] = useState(false);
  const [uploadState, setUploadState] = useState<UploadState>({
    status: "idle"
  });
  const [selectedTarget, setSelectedTarget] = useState("");
  const [confirmedTarget, setConfirmedTarget] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const csvInputRef = useRef<HTMLInputElement>(null);
  const toasts = useKumoToastManager();
  const [mcpState, setMcpState] = useState<MCPServersState>({
    prompts: [],
    resources: [],
    servers: {},
    tools: []
  });
  const [showMcpPanel, setShowMcpPanel] = useState(false);
  const [mcpName, setMcpName] = useState("");
  const [mcpUrl, setMcpUrl] = useState("");
  const [isAddingServer, setIsAddingServer] = useState(false);
  const mcpPanelRef = useRef<HTMLDivElement>(null);

  const agent = useAgent<ChatAgent>({
    agent: "ChatAgent",
    onOpen: useCallback(() => setConnected(true), []),
    onClose: useCallback(() => setConnected(false), []),
    onError: useCallback(
      (error: Event) => console.error("WebSocket error:", error),
      []
    ),
    onMcpUpdate: useCallback((state: MCPServersState) => {
      setMcpState(state);
    }, []),
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
          // Not JSON or not our event
        }
      },
      [toasts]
    )
  });

  // Close MCP panel when clicking outside
  useEffect(() => {
    if (!showMcpPanel) return;
    function handleClickOutside(e: MouseEvent) {
      if (
        mcpPanelRef.current &&
        !mcpPanelRef.current.contains(e.target as Node)
      ) {
        setShowMcpPanel(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showMcpPanel]);

  const handleAddServer = async () => {
    if (!mcpName.trim() || !mcpUrl.trim()) return;
    setIsAddingServer(true);
    try {
      await agent.stub.addServer(mcpName.trim(), mcpUrl.trim());
      setMcpName("");
      setMcpUrl("");
    } catch (e) {
      console.error("Failed to add MCP server:", e);
    } finally {
      setIsAddingServer(false);
    }
  };

  const handleRemoveServer = async (serverId: string) => {
    try {
      await agent.stub.removeServer(serverId);
    } catch (e) {
      console.error("Failed to remove MCP server:", e);
    }
  };

  const serverEntries = Object.entries(mcpState.servers);
  const mcpToolCount = mcpState.tools.length;

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
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Re-focus the input after streaming ends
  useEffect(() => {
    if (!isStreaming && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isStreaming]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const images = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (images.length === 0) return;
    setAttachments((prev) => [...prev, ...images.map(createAttachment)]);
  }, []);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => {
      const att = prev.find((a) => a.id === id);
      if (att) URL.revokeObjectURL(att.preview);
      return prev.filter((a) => a.id !== id);
    });
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.types.includes("Files")) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
    },
    [addFiles]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      const files: File[] = [];
      for (const item of items) {
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (file) files.push(file);
        }
      }
      if (files.length > 0) {
        e.preventDefault();
        addFiles(files);
      }
    },
    [addFiles]
  );

  const send = useCallback(async () => {
    const text = input.trim();
    if ((!text && attachments.length === 0) || isStreaming) return;
    setInput("");

    const parts: Array<
      | { type: "text"; text: string }
      | { type: "file"; mediaType: string; url: string }
    > = [];
    if (text) parts.push({ type: "text", text });

    for (const att of attachments) {
      const dataUri = await fileToDataUri(att.file);
      parts.push({ type: "file", mediaType: att.mediaType, url: dataUri });
    }

    for (const att of attachments) URL.revokeObjectURL(att.preview);
    setAttachments([]);

    sendMessage({ role: "user", parts });
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }, [input, attachments, isStreaming, sendMessage]);

  const resetWorkspace = useCallback(() => {
    setUploadState({ status: "idle" });
    setSelectedTarget("");
    setConfirmedTarget("");
    if (csvInputRef.current) csvInputRef.current.value = "";
  }, []);

  const handleCsvFile = useCallback(
    async (file: File) => {
      if (uploadState.status === "ready") {
        const replace = window.confirm(
          "Uploading a new CSV will replace the active dataset and clear the selected target."
        );
        if (!replace) return;
        setSelectedTarget("");
        setConfirmedTarget("");
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
        setSelectedTarget("");
        setConfirmedTarget("");
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

  const currentSummary =
    uploadState.status === "ready" ? uploadState.summary : null;
  const canConfirmTarget = Boolean(currentSummary && selectedTarget);

  return (
    <div
      className="flex flex-col h-screen bg-kumo-elevated relative"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-kumo-elevated/80 backdrop-blur-sm border-2 border-dashed border-kumo-brand rounded-xl m-2 pointer-events-none">
          <div className="flex flex-col items-center gap-2 text-kumo-brand">
            <ImageIcon size={40} />
            <Text variant="heading3">Drop images here</Text>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="px-5 py-4 bg-kumo-base border-b border-kumo-line">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-kumo-default">
              Automated FE
            </h1>
            <Badge variant="secondary">
              <ChatCircleDotsIcon size={12} weight="bold" className="mr-1" />
              Workspace
            </Badge>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <CircleIcon
                size={8}
                weight="fill"
                className={connected ? "text-kumo-success" : "text-kumo-danger"}
              />
              <Text size="xs" variant="secondary">
                {connected ? "Connected" : "Disconnected"}
              </Text>
            </div>
            <div className="flex items-center gap-1.5">
              <BugIcon size={14} className="text-kumo-inactive" />
              <Switch
                checked={showDebug}
                onCheckedChange={setShowDebug}
                size="sm"
                aria-label="Toggle debug mode"
              />
            </div>
            <ThemeToggle />
            <div className="relative" ref={mcpPanelRef}>
              <Button
                variant="secondary"
                icon={<PlugsConnectedIcon size={16} />}
                onClick={() => setShowMcpPanel(!showMcpPanel)}
              >
                MCP
                {mcpToolCount > 0 && (
                  <Badge variant="primary" className="ml-1.5">
                    <WrenchIcon size={10} className="mr-0.5" />
                    {mcpToolCount}
                  </Badge>
                )}
              </Button>

              {/* MCP Dropdown Panel */}
              {showMcpPanel && (
                <div className="absolute right-0 top-full mt-2 w-96 z-50">
                  <Surface className="rounded-xl ring ring-kumo-line shadow-lg p-4 space-y-4">
                    {/* Panel Header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <PlugsConnectedIcon
                          size={16}
                          className="text-kumo-accent"
                        />
                        <Text size="sm" bold>
                          MCP Servers
                        </Text>
                        {serverEntries.length > 0 && (
                          <Badge variant="secondary">
                            {serverEntries.length}
                          </Badge>
                        )}
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        shape="square"
                        aria-label="Close MCP panel"
                        icon={<XIcon size={14} />}
                        onClick={() => setShowMcpPanel(false)}
                      />
                    </div>

                    {/* Add Server Form */}
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        handleAddServer();
                      }}
                      className="space-y-2"
                    >
                      <input
                        type="text"
                        value={mcpName}
                        onChange={(e) => setMcpName(e.target.value)}
                        placeholder="Server name"
                        className="w-full px-3 py-1.5 text-sm rounded-lg border border-kumo-line bg-kumo-base text-kumo-default placeholder:text-kumo-inactive focus:outline-none focus:ring-1 focus:ring-kumo-accent"
                      />
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={mcpUrl}
                          onChange={(e) => setMcpUrl(e.target.value)}
                          placeholder="https://mcp.example.com"
                          className="flex-1 px-3 py-1.5 text-sm rounded-lg border border-kumo-line bg-kumo-base text-kumo-default placeholder:text-kumo-inactive focus:outline-none focus:ring-1 focus:ring-kumo-accent font-mono"
                        />
                        <Button
                          type="submit"
                          variant="primary"
                          size="sm"
                          icon={<PlusIcon size={14} />}
                          disabled={
                            isAddingServer || !mcpName.trim() || !mcpUrl.trim()
                          }
                        >
                          {isAddingServer ? "..." : "Add"}
                        </Button>
                      </div>
                    </form>

                    {/* Server List */}
                    {serverEntries.length > 0 && (
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {serverEntries.map(([id, server]) => (
                          <div
                            key={id}
                            className="flex items-start justify-between p-2.5 rounded-lg border border-kumo-line"
                          >
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-kumo-default truncate">
                                  {server.name}
                                </span>
                                <Badge
                                  variant={
                                    server.state === "ready"
                                      ? "primary"
                                      : server.state === "failed"
                                        ? "destructive"
                                        : "secondary"
                                  }
                                >
                                  {server.state}
                                </Badge>
                              </div>
                              <span className="text-xs font-mono text-kumo-subtle truncate block mt-0.5">
                                {server.server_url}
                              </span>
                              {server.state === "failed" && server.error && (
                                <span className="text-xs text-red-500 block mt-0.5">
                                  {server.error}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-1 shrink-0 ml-2">
                              {server.state === "authenticating" &&
                                server.auth_url && (
                                  <Button
                                    variant="primary"
                                    size="sm"
                                    icon={<SignInIcon size={12} />}
                                    onClick={() =>
                                      window.open(
                                        server.auth_url as string,
                                        "oauth",
                                        "width=600,height=800"
                                      )
                                    }
                                  >
                                    Auth
                                  </Button>
                                )}
                              <Button
                                variant="ghost"
                                size="sm"
                                shape="square"
                                aria-label="Remove server"
                                icon={<TrashIcon size={12} />}
                                onClick={() => handleRemoveServer(id)}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Tool Summary */}
                    {mcpToolCount > 0 && (
                      <div className="pt-2 border-t border-kumo-line">
                        <div className="flex items-center gap-2">
                          <WrenchIcon size={14} className="text-kumo-subtle" />
                          <span className="text-xs text-kumo-subtle">
                            {mcpToolCount} tool
                            {mcpToolCount !== 1 ? "s" : ""} available from MCP
                            servers
                          </span>
                        </div>
                      </div>
                    )}
                  </Surface>
                </div>
              )}
            </div>
            <Button
              variant="secondary"
              icon={<TrashIcon size={16} />}
              onClick={clearHistory}
            >
              Clear
            </Button>
          </div>
        </div>
      </header>

      {/* Active workspace */}
      <section className="border-b border-kumo-line bg-kumo-base">
        <div className="max-w-5xl mx-auto px-5 py-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
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
                    Upload one CSV file to inspect columns and choose the target
                    variable.
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
                  `.csv` only, up to {formatBytes(MAX_CSV_SIZE_BYTES)}. No LLM
                  calls run during upload or target selection.
                </Text>
              </div>

              {uploadState.status === "validating" && (
                <Badge variant="secondary">
                  Validating {uploadState.fileName}
                </Badge>
              )}
              {uploadState.status === "parsing" && (
                <Badge variant="secondary">
                  Parsing {uploadState.fileName}
                </Badge>
              )}
              {uploadState.status === "error" && (
                <div className="rounded-lg border border-kumo-danger/40 bg-kumo-danger/10 px-3 py-2">
                  <Text size="sm">{uploadState.message}</Text>
                </div>
              )}

              {currentSummary && (
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

                  <div className="overflow-auto rounded-lg border border-kumo-line">
                    <table className="min-w-full text-left text-sm">
                      <thead className="bg-kumo-elevated text-kumo-subtle">
                        <tr>
                          <th className="px-3 py-2 font-medium">Column</th>
                          <th className="px-3 py-2 font-medium">Type</th>
                          <th className="px-3 py-2 font-medium">Missing</th>
                          <th className="px-3 py-2 font-medium">Samples</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-kumo-line">
                        {currentSummary.columns.map((column) => (
                          <tr key={column.name}>
                            <td className="px-3 py-2 font-medium text-kumo-default">
                              {column.name}
                              {confirmedTarget === column.name && (
                                <Badge variant="primary" className="ml-2">
                                  Target
                                </Badge>
                              )}
                            </td>
                            <td className="px-3 py-2 text-kumo-subtle">
                              {column.inferredType}
                            </td>
                            <td className="px-3 py-2 text-kumo-subtle">
                              {column.missingCount}
                            </td>
                            <td className="px-3 py-2 text-kumo-subtle">
                              {column.sampleValues.length > 0
                                ? column.sampleValues.join(", ")
                                : "No non-empty samples"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <details className="rounded-lg border border-kumo-line bg-kumo-elevated">
                    <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-kumo-default">
                      Preview first {currentSummary.previewRows.length} rows
                    </summary>
                    <div className="overflow-auto border-t border-kumo-line">
                      <table className="min-w-full text-left text-xs">
                        <thead className="bg-kumo-base text-kumo-subtle">
                          <tr>
                            {currentSummary.columns.map((column) => (
                              <th
                                key={column.name}
                                className="px-3 py-2 font-medium"
                              >
                                {column.name}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-kumo-line">
                          {currentSummary.previewRows.map((row, rowIndex) => (
                            <tr key={rowIndex}>
                              {currentSummary.columns.map((column) => (
                                <td
                                  key={column.name}
                                  className="max-w-52 truncate px-3 py-2 text-kumo-subtle"
                                >
                                  {row[column.name] == null
                                    ? "NULL"
                                    : String(row[column.name])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                </div>
              )}
            </div>
          </Surface>

          <Surface className="rounded-xl ring ring-kumo-line p-4">
            <div className="space-y-4">
              <div>
                <Text size="lg" bold>
                  Target variable
                </Text>
                <Text size="sm" variant="secondary">
                  Select the prediction target after the CSV is parsed.
                </Text>
              </div>

              <select
                value={selectedTarget}
                onChange={(event) => setSelectedTarget(event.target.value)}
                disabled={!currentSummary}
                className="w-full rounded-lg border border-kumo-line bg-kumo-base px-3 py-2 text-sm text-kumo-default focus:outline-none focus:ring-2 focus:ring-kumo-ring disabled:cursor-not-allowed disabled:opacity-60"
              >
                <option value="">
                  {currentSummary ? "Choose a column" : "Upload a CSV first"}
                </option>
                {currentSummary?.columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </select>

              <Button
                type="button"
                variant="primary"
                icon={<CheckCircleIcon size={14} />}
                disabled={!canConfirmTarget}
                onClick={() => {
                  setConfirmedTarget(selectedTarget);
                  toasts.add({
                    title: "Target selected",
                    description: selectedTarget
                  });
                }}
              >
                Confirm target
              </Button>

              {confirmedTarget ? (
                <div className="rounded-lg border border-kumo-success/40 bg-kumo-success/10 px-3 py-2">
                  <Text size="sm" bold>
                    {confirmedTarget}
                  </Text>
                  <Text size="xs" variant="secondary">
                    The workspace is ready for split setup and later feature
                    planning.
                  </Text>
                </div>
              ) : (
                <div className="rounded-lg border border-kumo-line bg-kumo-elevated px-3 py-2">
                  <Text size="xs" variant="secondary">
                    No target selected. LLM analysis remains disabled for this
                    milestone.
                  </Text>
                </div>
              )}
            </div>
          </Surface>
        </div>
      </section>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-5 py-6 space-y-5">
          {messages.length === 0 && (
            <Empty
              icon={<ChatCircleDotsIcon size={32} />}
              title="Start a conversation"
              contents={
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    "What's the weather in Paris?",
                    "What timezone am I in?",
                    "Calculate 5000 * 3",
                    "Remind me in 5 minutes to take a break"
                  ].map((prompt) => (
                    <Button
                      key={prompt}
                      variant="outline"
                      size="sm"
                      disabled={isStreaming}
                      onClick={() => {
                        sendMessage({
                          role: "user",
                          parts: [{ type: "text", text: prompt }]
                        });
                      }}
                    >
                      {prompt}
                    </Button>
                  ))}
                </div>
              }
            />
          )}

          {messages.map((message: UIMessage, index: number) => {
            const isUser = message.role === "user";
            const isLastAssistant =
              message.role === "assistant" && index === messages.length - 1;

            return (
              <div key={message.id} className="space-y-2">
                {showDebug && (
                  <pre className="text-[11px] text-kumo-subtle bg-kumo-control rounded-lg p-3 overflow-auto max-h-64">
                    {JSON.stringify(message, null, 2)}
                  </pre>
                )}

                {/* Tool parts */}
                {message.parts.filter(isToolUIPart).map((part) => (
                  <ToolPartView
                    key={part.toolCallId}
                    part={part}
                    addToolApprovalResponse={addToolApprovalResponse}
                  />
                ))}

                {/* Reasoning parts */}
                {message.parts
                  .filter(
                    (part) =>
                      part.type === "reasoning" &&
                      (part as { text?: string }).text?.trim()
                  )
                  .map((part, i) => {
                    const reasoning = part as {
                      type: "reasoning";
                      text: string;
                      state?: "streaming" | "done";
                    };
                    const isDone = reasoning.state === "done" || !isStreaming;
                    return (
                      <div key={i} className="flex justify-start">
                        <details className="max-w-[85%] w-full" open={!isDone}>
                          <summary className="flex items-center gap-2 cursor-pointer px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-sm select-none">
                            <BrainIcon size={14} className="text-purple-400" />
                            <span className="font-medium text-kumo-default">
                              Reasoning
                            </span>
                            {isDone ? (
                              <span className="text-xs text-kumo-success">
                                Complete
                              </span>
                            ) : (
                              <span className="text-xs text-kumo-brand">
                                Thinking...
                              </span>
                            )}
                            <CaretDownIcon
                              size={14}
                              className="ml-auto text-kumo-inactive"
                            />
                          </summary>
                          <pre className="mt-2 px-3 py-2 rounded-lg bg-kumo-control text-xs text-kumo-default whitespace-pre-wrap overflow-auto max-h-64">
                            {reasoning.text}
                          </pre>
                        </details>
                      </div>
                    );
                  })}

                {/* Image parts */}
                {message.parts
                  .filter(
                    (part): part is Extract<typeof part, { type: "file" }> =>
                      part.type === "file" &&
                      (part as { mediaType?: string }).mediaType?.startsWith(
                        "image/"
                      ) === true
                  )
                  .map((part, i) => (
                    <div
                      key={`file-${i}`}
                      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      <img
                        src={part.url}
                        alt="Attachment"
                        className="max-h-64 rounded-xl border border-kumo-line object-contain"
                      />
                    </div>
                  ))}

                {/* Text parts */}
                {message.parts
                  .filter((part) => part.type === "text")
                  .map((part, i) => {
                    const text = (part as { type: "text"; text: string }).text;
                    if (!text) return null;

                    if (isUser) {
                      return (
                        <div key={i} className="flex justify-end">
                          <div className="max-w-[85%] px-4 py-2.5 rounded-2xl rounded-br-md bg-kumo-contrast text-kumo-inverse leading-relaxed">
                            {text}
                          </div>
                        </div>
                      );
                    }

                    return (
                      <div key={i} className="flex justify-start">
                        <div className="max-w-[85%] rounded-2xl rounded-bl-md bg-kumo-base text-kumo-default leading-relaxed">
                          <Streamdown
                            className="sd-theme rounded-2xl rounded-bl-md p-3"
                            plugins={{ code }}
                            controls={false}
                            isAnimating={isLastAssistant && isStreaming}
                          >
                            {text}
                          </Streamdown>
                        </div>
                      </div>
                    );
                  })}
              </div>
            );
          })}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-kumo-line bg-kumo-base">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="max-w-3xl mx-auto px-5 py-4"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {attachments.length > 0 && (
            <div className="flex gap-2 mb-2 flex-wrap">
              {attachments.map((att) => (
                <div
                  key={att.id}
                  className="relative group rounded-lg border border-kumo-line bg-kumo-control overflow-hidden"
                >
                  <img
                    src={att.preview}
                    alt={att.file.name}
                    className="h-16 w-16 object-cover"
                  />
                  <button
                    type="button"
                    onClick={() => removeAttachment(att.id)}
                    className="absolute top-0.5 right-0.5 rounded-full bg-kumo-contrast/80 text-kumo-inverse p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                    aria-label={`Remove ${att.file.name}`}
                  >
                    <XIcon size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex items-end gap-3 rounded-xl border border-kumo-line bg-kumo-base p-3 shadow-sm focus-within:ring-2 focus-within:ring-kumo-ring focus-within:border-transparent transition-shadow">
            <Button
              type="button"
              variant="ghost"
              shape="square"
              aria-label="Attach images"
              icon={<PaperclipIcon size={18} />}
              onClick={() => fileInputRef.current?.click()}
              disabled={!connected || isStreaming}
              className="mb-0.5"
            />
            <InputArea
              ref={textareaRef}
              value={input}
              onValueChange={setInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${el.scrollHeight}px`;
              }}
              onPaste={handlePaste}
              placeholder={
                attachments.length > 0
                  ? "Add a message or send images..."
                  : "Send a message..."
              }
              disabled={!connected || isStreaming}
              rows={1}
              className="flex-1 ring-0! focus:ring-0! shadow-none! bg-transparent! outline-none! resize-none max-h-40"
            />
            {isStreaming ? (
              <Button
                type="button"
                variant="secondary"
                shape="square"
                aria-label="Stop generation"
                icon={<StopIcon size={18} />}
                onClick={stop}
                className="mb-0.5"
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
                icon={<PaperPlaneRightIcon size={18} />}
                className="mb-0.5"
              />
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Toasty>
      <Suspense
        fallback={
          <div className="flex items-center justify-center h-screen text-kumo-inactive">
            Loading...
          </div>
        }
      >
        <Chat />
      </Suspense>
    </Toasty>
  );
}
