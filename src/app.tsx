import { useCallback, useRef, useState } from "react";
import {
  MAX_CSV_SIZE_BYTES,
  applyPreprocessingStepsToPreviewRows,
  describePreprocessingStep,
  formatBytes,
  parseCsvFile,
  renderPreviewValue,
  validateCsvFile,
  type DatasetSummary,
  type PreprocessingStep,
  type PreprocessingStatus
} from "./csv-profile";
import { Badge, Button, Empty, Surface, Text } from "@cloudflare/kumo";
import { Toasty, useKumoToastManager } from "@cloudflare/kumo/components/toast";
import {
  CheckCircleIcon,
  MoonIcon,
  PaperclipIcon,
  SunIcon,
  TrashIcon
} from "@phosphor-icons/react";

type UploadState =
  | { status: "idle" }
  | { status: "validating"; fileName: string }
  | { status: "parsing"; fileName: string }
  | { status: "ready"; summary: DatasetSummary }
  | { status: "error"; message: string };

const PREVIEW_VISIBLE_COLUMN_COUNT = 10;

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

  const updatePreprocessingStatus = useCallback(
    (stepId: string, status: PreprocessingStatus) => {
      setPreprocessingStatuses((current) => ({
        ...current,
        [stepId]: status
      }));
    },
    []
  );

  const resetWorkspace = useCallback(() => {
    setUploadState({ status: "idle" });
    setPreprocessingStatuses({});
    if (csvInputRef.current) csvInputRef.current.value = "";
  }, []);

  const handleCsvFile = useCallback(
    async (file: File) => {
      if (uploadState.status === "ready") {
        const replace = window.confirm(
          "Uploading a new CSV will replace the active dataset and preprocessing pipeline."
        );
        if (!replace) return;
        setPreprocessingStatuses({});
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

            <div className="overflow-auto rounded-lg border border-kumo-line">
              <table className="min-w-full text-left text-sm">
                <thead className="bg-kumo-elevated text-kumo-subtle">
                  <tr>
                    <th className="px-3 py-2 font-medium">Column</th>
                    <th className="px-3 py-2 font-medium">Type</th>
                    <th className="px-3 py-2 font-medium">Missing</th>
                    <th className="px-3 py-2 font-medium">Samples</th>
                    <th className="px-3 py-2 font-medium">Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-kumo-line">
                  {currentSummary.columns.map((column) => (
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
        <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
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

      <main className="mx-auto max-w-5xl px-5 py-5">
        <DatasetWorkspace />
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
