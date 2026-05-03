import Papa from "papaparse";

export const MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024;
const MAX_INFERENCE_ROWS = 1000;
const MAX_PREVIEW_ROWS = 20;
const HIGH_MISSINGNESS_THRESHOLD = 0.3;
const LIKELY_IDENTIFIER_UNIQUE_RATIO = 0.95;
const MIN_IDENTIFIER_ROW_COUNT = 10;

export type InferredColumnType =
  | "string"
  | "number"
  | "boolean"
  | "date"
  | "empty"
  | "mixed";

export type PreviewValue = string | number | boolean | null;

export type ProfilingNoteCode =
  | "leading_trailing_whitespace"
  | "case_variants"
  | "numeric_after_trim"
  | "boolean_after_case_fold"
  | "date_after_trim"
  | "missing_like_tokens"
  | "high_missingness"
  | "constant_column"
  | "likely_identifier"
  | "empty_column";

export type PreprocessingStatus = "proposed" | "accepted" | "rejected";

export interface ProfilingNote {
  code: ProfilingNoteCode;
  message: string;
  affectedCount: number;
}

export type PreprocessingStep =
  | {
      id: string;
      operation: "trim_whitespace";
      columns: string[];
      status: PreprocessingStatus;
    }
  | {
      id: string;
      operation: "normalize_boolean";
      column: string;
      trueValues: string[];
      falseValues: string[];
      status: PreprocessingStatus;
    }
  | {
      id: string;
      operation: "standardize_missing_tokens";
      columns: string[];
      tokens: string[];
      status: PreprocessingStatus;
    };

export interface ColumnSummary {
  name: string;
  inferredType: InferredColumnType;
  missingCount: number;
  nonMissingCount: number;
  uniqueCount: number;
  uniqueRatio: number;
  topValues: { value: string; count: number; percent: number }[];
  sampleValues: string[];
  profilingNotes: ProfilingNote[];
}

export interface DatasetSummary {
  fileName: string;
  fileSizeBytes: number;
  parsedRowCount: number;
  columns: ColumnSummary[];
  previewRows: Record<string, PreviewValue>[];
  warnings: string[];
  proposedPreprocessingSteps: PreprocessingStep[];
}

export type ColumnAssumptionRole =
  | "feature"
  | "target"
  | "identifier"
  | "timestamp"
  | "free_text"
  | "ignore"
  | "unknown";

export type ColumnAssumptionSemanticType =
  | "numeric"
  | "categorical"
  | "boolean"
  | "date"
  | "text"
  | "id"
  | "mixed"
  | "empty";

export type ColumnAssumptionConfidence = "low" | "medium" | "high";

export interface ColumnAssumption {
  id: string;
  columnName: string;
  role: ColumnAssumptionRole;
  semanticType: ColumnAssumptionSemanticType;
  confidence: ColumnAssumptionConfidence;
  evidence: string[];
  risks: string[];
  recommendedActions: string[];
  status: PreprocessingStatus;
}

export interface IntentOverride {
  assumptionId: string;
  columnName: string;
  role: ColumnAssumptionRole;
  status: PreprocessingStatus;
  source: "user";
  reason: string;
  updatedAt: string;
}

export interface DatasetIntent {
  targetColumn?: string;
  ignoredColumns: string[];
  identifierColumns: string[];
  timestampColumns: string[];
  textColumns: string[];
  featureColumns: string[];
  unknownColumns: string[];
  acceptedAssumptions: ColumnAssumption[];
  rejectedAssumptionIds: string[];
  userOverrides: IntentOverride[];
  conflicts: string[];
  warnings: string[];
}

export type TransformationDecisionType =
  | "assumption"
  | "mapping"
  | "normalization"
  | "validation"
  | "exclusion";

export interface TransformationDecision {
  id: string;
  type: TransformationDecisionType;
  target: string;
  decision: string;
  reason: string;
  confidence: ColumnAssumptionConfidence;
  evidence: string[];
  alternatives: string[];
  relatedAssumptionIds: string[];
  relatedPreprocessingStepIds: string[];
}

export type PreprocessingSuggestionAction =
  | "none"
  | "drop"
  | "fill_mean"
  | "fill_median"
  | "fill_mode"
  | "one_hot_encode"
  | "trim_whitespace"
  | "standardize_missing"
  | "normalize_boolean"
  | "split_name";

export interface PreprocessingSuggestion {
  columnName: string;
  action: PreprocessingSuggestionAction;
  reason: string;
  alternatives: PreprocessingSuggestionAction[];
}

export type ColumnSelectionDecision = "keep" | "drop" | "review";

export interface TargetSuggestion {
  columnName: string;
  reason: string;
  confidence: ColumnAssumptionConfidence;
}

export interface ColumnDecision {
  columnName: string;
  decision: ColumnSelectionDecision;
  reason: string;
  confidence: ColumnAssumptionConfidence;
  alternatives: ColumnSelectionDecision[];
}

export interface ColumnSelectionPlan {
  datasetSummary: string;
  targetSuggestion?: TargetSuggestion;
  columnDecisions: ColumnDecision[];
  globalWarnings: string[];
  nextQuestions: string[];
}

export interface ColumnPreprocessingPlan {
  preprocessingSuggestions: PreprocessingSuggestion[];
  globalWarnings: string[];
  nextQuestions: string[];
}

export interface LlmPreprocessingPlan {
  datasetSummary: string;
  assumptions: ColumnAssumption[];
  preprocessingSuggestions: PreprocessingSuggestion[];
  decisions: TransformationDecision[];
  globalWarnings: string[];
  nextQuestions: string[];
}

export interface AiReviewProfileColumn {
  name: string;
  inferredType: InferredColumnType;
  missingCount: number;
  missingPercent: number;
  nonMissingCount: number;
  uniqueCount: number;
  uniqueRatio: number;
  topValues: { value: string; count: number; percent: number }[];
  sampleValues: string[];
  profilingNotes: ProfilingNote[];
}

export interface AiReviewProfile {
  fileName: string;
  fileSizeBytes: number;
  parsedRowCount: number;
  columnCount: number;
  warnings: string[];
  columns: AiReviewProfileColumn[];
  proposedPreprocessingSteps: Array<{
    id: string;
    operation: PreprocessingStep["operation"];
    target: string;
    description: string;
  }>;
  previewRows: Record<string, PreviewValue>[];
}

export function buildAiReviewProfile(summary: DatasetSummary): AiReviewProfile {
  return {
    fileName: summary.fileName,
    fileSizeBytes: summary.fileSizeBytes,
    parsedRowCount: summary.parsedRowCount,
    columnCount: summary.columns.length,
    warnings: summary.warnings,
    columns: summary.columns.map((column) => ({
      name: column.name,
      inferredType: column.inferredType,
      missingCount: column.missingCount,
      missingPercent:
        summary.parsedRowCount === 0
          ? 0
          : (column.missingCount / summary.parsedRowCount) * 100,
      nonMissingCount: column.nonMissingCount,
      uniqueCount: column.uniqueCount,
      uniqueRatio: column.uniqueRatio,
      topValues: column.topValues.slice(0, 5),
      sampleValues: column.sampleValues.slice(0, 5),
      profilingNotes: column.profilingNotes
    })),
    proposedPreprocessingSteps: summary.proposedPreprocessingSteps.map(
      (step) => {
        const description = describePreprocessingStep(step);
        return {
          id: step.id,
          operation: step.operation,
          target:
            step.operation === "normalize_boolean"
              ? step.column
              : step.columns.join(", "),
          description: `${description.title}: ${description.description}`
        };
      }
    ),
    previewRows: summary.previewRows.slice(0, 20)
  };
}

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values)).sort((left, right) =>
    left.localeCompare(right)
  );
}

function addRoleColumn(
  intent: DatasetIntent,
  role: ColumnAssumptionRole,
  columnName: string
) {
  switch (role) {
    case "target":
      intent.targetColumn = intent.targetColumn ?? columnName;
      break;
    case "identifier":
      intent.identifierColumns.push(columnName);
      break;
    case "timestamp":
      intent.timestampColumns.push(columnName);
      break;
    case "free_text":
      intent.textColumns.push(columnName);
      break;
    case "ignore":
      intent.ignoredColumns.push(columnName);
      break;
    case "feature":
      intent.featureColumns.push(columnName);
      break;
    case "unknown":
      intent.unknownColumns.push(columnName);
      break;
  }
}

function pushRoleConflict(
  conflicts: string[],
  columnName: string,
  roles: ColumnAssumptionRole[]
) {
  conflicts.push(
    `${columnName} has multiple accepted roles: ${roles.join(", ")}. Keep only the role that matches the dataset objective.`
  );
}

export function buildDatasetIntent(
  summary: DatasetSummary,
  assumptions: ColumnAssumption[],
  userOverrides: IntentOverride[] = []
): DatasetIntent {
  const acceptedAssumptions = assumptions.filter(
    (assumption) => assumption.status === "accepted"
  );
  const rejectedAssumptionIds = assumptions
    .filter((assumption) => assumption.status === "rejected")
    .map((assumption) => assumption.id);
  const intent: DatasetIntent = {
    ignoredColumns: [],
    identifierColumns: [],
    timestampColumns: [],
    textColumns: [],
    featureColumns: [],
    unknownColumns: [],
    acceptedAssumptions,
    rejectedAssumptionIds,
    userOverrides,
    conflicts: [],
    warnings: []
  };
  const rolesByColumn = new Map<string, Set<ColumnAssumptionRole>>();
  const targetCandidates = acceptedAssumptions.filter(
    (assumption) => assumption.role === "target"
  );

  for (const assumption of acceptedAssumptions) {
    addRoleColumn(intent, assumption.role, assumption.columnName);
    const roles = rolesByColumn.get(assumption.columnName) ?? new Set();
    roles.add(assumption.role);
    rolesByColumn.set(assumption.columnName, roles);
  }

  if (targetCandidates.length > 1) {
    intent.conflicts.push(
      `Multiple target columns are accepted: ${targetCandidates.map((assumption) => assumption.columnName).join(", ")}. Choose one target before generating modeling steps.`
    );
  }

  if (targetCandidates.length === 1) {
    intent.targetColumn = targetCandidates[0].columnName;
  }

  for (const [columnName, roles] of rolesByColumn) {
    if (roles.size > 1)
      pushRoleConflict(intent.conflicts, columnName, [...roles]);
  }

  const excludedFeatureColumns = new Set([
    ...intent.identifierColumns,
    ...intent.ignoredColumns,
    ...(intent.targetColumn ? [intent.targetColumn] : [])
  ]);
  intent.featureColumns = intent.featureColumns.filter(
    (columnName) => !excludedFeatureColumns.has(columnName)
  );

  for (const column of summary.columns) {
    if (
      column.inferredType === "empty" &&
      (intent.featureColumns.includes(column.name) ||
        intent.targetColumn === column.name)
    ) {
      intent.conflicts.push(
        `${column.name} is empty but accepted as ${intent.targetColumn === column.name ? "target" : "feature"}.`
      );
    }
  }

  if (!intent.targetColumn) {
    intent.warnings.push(
      "No target column has been accepted yet. Downstream modeling recommendations will remain incomplete."
    );
  }

  if (intent.featureColumns.length === 0) {
    intent.warnings.push(
      "No feature columns have been accepted yet. Accept feature assumptions or confirm columns manually before generating modeling steps."
    );
  }

  return {
    ...intent,
    ignoredColumns: uniqueSorted(intent.ignoredColumns),
    identifierColumns: uniqueSorted(intent.identifierColumns),
    timestampColumns: uniqueSorted(intent.timestampColumns),
    textColumns: uniqueSorted(intent.textColumns),
    featureColumns: uniqueSorted(intent.featureColumns),
    unknownColumns: uniqueSorted(intent.unknownColumns),
    rejectedAssumptionIds: uniqueSorted(intent.rejectedAssumptionIds)
  };
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export function validateCsvFile(file: File): string | null {
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

function isMissing(value: unknown) {
  return value == null || String(value).trim() === "";
}

function isMissingLikeToken(value: string) {
  return /^(n\/a|na|null|none|nil|-|--|\?)$/i.test(value.trim());
}

function looksBoolean(value: string) {
  return /^(true|false|0|1|yes|no)$/i.test(value.trim());
}

function looksNumber(value: string) {
  if (value.trim() === "") return false;
  return Number.isFinite(Number(value));
}

function looksDate(value: string) {
  const trimmed = value.trim();
  if (!trimmed || looksNumber(trimmed)) return false;
  const timestamp = Date.parse(trimmed);
  return Number.isFinite(timestamp);
}

function inferColumnType(values: string[]): InferredColumnType {
  const present = values.filter((value) => !isMissing(value));
  if (present.length === 0) return "empty";

  const checks: Array<[InferredColumnType, (value: string) => boolean]> = [
    ["number", looksNumber],
    ["boolean", looksBoolean],
    ["date", looksDate]
  ];

  for (const [type, check] of checks) {
    if (present.every(check)) return type;
  }

  const hasStructuredSignal = present.some(
    (value) => looksNumber(value) || looksBoolean(value) || looksDate(value)
  );
  return hasStructuredSignal ? "mixed" : "string";
}

function normalizePreviewValue(value: unknown): PreviewValue {
  if (value == null) return null;
  return String(value);
}

export function describePreprocessingStep(step: PreprocessingStep) {
  switch (step.operation) {
    case "trim_whitespace":
      return {
        title: "Trim whitespace",
        description: `Remove leading and trailing whitespace from ${step.columns.length} column${step.columns.length === 1 ? "" : "s"}.`
      };
    case "normalize_boolean":
      return {
        title: "Normalize boolean values",
        description: `Standardize boolean-like values in ${step.column}.`
      };
    case "standardize_missing_tokens":
      return {
        title: "Standardize missing tokens",
        description: `Treat tokens like ${step.tokens.join(", ")} as missing values in ${step.columns.length} column${step.columns.length === 1 ? "" : "s"}.`
      };
  }
}

export function renderPreviewValue(value: PreviewValue) {
  if (value == null) return "NULL";
  const text = String(value);
  if (text === "") return '""';
  if (text !== text.trim()) return JSON.stringify(text);
  return text;
}

function standardizeBooleanValue(value: string, step: PreprocessingStep) {
  if (step.operation !== "normalize_boolean") return value;
  const normalized = value.trim().toLowerCase();
  if (step.trueValues.includes(normalized)) return "true";
  if (step.falseValues.includes(normalized)) return "false";
  return value;
}

function standardizeMissingValue(value: string, step: PreprocessingStep) {
  if (step.operation !== "standardize_missing_tokens") return value;
  const normalizedTokens = new Set(
    step.tokens.map((token) => token.trim().toLowerCase())
  );
  return normalizedTokens.has(value.trim().toLowerCase()) ? null : value;
}

export function applyPreprocessingStepsToPreviewRows(
  rows: Record<string, PreviewValue>[],
  steps: PreprocessingStep[]
) {
  return rows.map((row) => {
    const nextRow: Record<string, PreviewValue> = { ...row };

    steps.forEach((step) => {
      switch (step.operation) {
        case "trim_whitespace":
          step.columns.forEach((column) => {
            const value = nextRow[column];
            if (typeof value === "string") nextRow[column] = value.trim();
          });
          break;
        case "normalize_boolean": {
          const value = nextRow[step.column];
          if (typeof value === "string") {
            nextRow[step.column] = standardizeBooleanValue(value, step);
          }
          break;
        }
        case "standardize_missing_tokens":
          step.columns.forEach((column) => {
            const value = nextRow[column];
            if (typeof value === "string") {
              nextRow[column] = standardizeMissingValue(value, step);
            }
          });
          break;
      }
    });

    return nextRow;
  });
}

export function buildProfilingNotes(
  field: string,
  values: string[]
): ProfilingNote[] {
  const notes: ProfilingNote[] = [];
  const missingCount = values.filter(isMissing).length;
  const presentValues = values.filter((value) => !isMissing(value));
  const uniqueCount = new Set(presentValues).size;
  const missingRatio = values.length === 0 ? 0 : missingCount / values.length;
  const whitespaceCount = values.filter(
    (value) => value.length > 0 && value !== value.trim()
  ).length;
  const numericAfterTrimCount = values.filter(
    (value) => value !== value.trim() && looksNumber(value.trim())
  ).length;
  const booleanAfterCaseFoldCount = values.filter((value) => {
    const trimmed = value.trim();
    return looksBoolean(trimmed) && trimmed !== trimmed.toLowerCase();
  }).length;
  const dateAfterTrimCount = values.filter(
    (value) => value !== value.trim() && looksDate(value.trim())
  ).length;
  const missingLikeTokenCount = values.filter(isMissingLikeToken).length;
  const caseGroups = new Map<string, Set<string>>();

  values.forEach((value) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    const lower = trimmed.toLowerCase();
    const variants = caseGroups.get(lower) ?? new Set<string>();
    variants.add(trimmed);
    caseGroups.set(lower, variants);
  });

  const caseVariantCount = values.filter((value) => {
    const trimmed = value.trim();
    if (!trimmed) return false;
    return (caseGroups.get(trimmed.toLowerCase())?.size ?? 0) > 1;
  }).length;

  if (whitespaceCount > 0) {
    notes.push({
      code: "leading_trailing_whitespace",
      message: `${field} contains leading or trailing whitespace. Raw preview keeps those values unchanged.`,
      affectedCount: whitespaceCount
    });
  }
  if (caseVariantCount > 0) {
    notes.push({
      code: "case_variants",
      message: `${field} contains values that differ only by letter case.`,
      affectedCount: caseVariantCount
    });
  }
  if (numericAfterTrimCount > 0) {
    notes.push({
      code: "numeric_after_trim",
      message: `${field} has numeric-like values after trimming whitespace.`,
      affectedCount: numericAfterTrimCount
    });
  }
  if (booleanAfterCaseFoldCount > 0) {
    notes.push({
      code: "boolean_after_case_fold",
      message: `${field} has boolean-like values after case normalization.`,
      affectedCount: booleanAfterCaseFoldCount
    });
  }
  if (dateAfterTrimCount > 0) {
    notes.push({
      code: "date_after_trim",
      message: `${field} has date-like values after trimming whitespace.`,
      affectedCount: dateAfterTrimCount
    });
  }
  if (missingLikeTokenCount > 0) {
    notes.push({
      code: "missing_like_tokens",
      message: `${field} contains tokens that commonly represent missing values.`,
      affectedCount: missingLikeTokenCount
    });
  }
  if (missingCount === values.length) {
    notes.push({
      code: "empty_column",
      message: `${field} has no non-empty values.`,
      affectedCount: missingCount
    });
  } else if (missingRatio >= HIGH_MISSINGNESS_THRESHOLD) {
    notes.push({
      code: "high_missingness",
      message: `${field} is missing at least ${Math.round(HIGH_MISSINGNESS_THRESHOLD * 100)}% of sampled rows.`,
      affectedCount: missingCount
    });
  }
  if (uniqueCount === 1 && presentValues.length > 0) {
    notes.push({
      code: "constant_column",
      message: `${field} has only one unique non-empty value.`,
      affectedCount: presentValues.length
    });
  }
  if (
    presentValues.length >= MIN_IDENTIFIER_ROW_COUNT &&
    uniqueCount / values.length >= LIKELY_IDENTIFIER_UNIQUE_RATIO
  ) {
    notes.push({
      code: "likely_identifier",
      message: `${field} is likely an identifier because most non-empty values are unique.`,
      affectedCount: uniqueCount
    });
  }

  return notes;
}

function buildPreprocessingSteps(
  columns: ColumnSummary[]
): PreprocessingStep[] {
  const columnsWith = (code: ProfilingNoteCode) =>
    columns
      .filter((column) =>
        column.profilingNotes.some((note) => note.code === code)
      )
      .map((column) => column.name);
  const steps: PreprocessingStep[] = [];
  const whitespaceColumns = columnsWith("leading_trailing_whitespace");
  const missingTokenColumns = columnsWith("missing_like_tokens");
  const booleanColumns = columnsWith("boolean_after_case_fold").filter(
    (columnName) =>
      columns.find((column) => column.name === columnName)?.inferredType ===
      "boolean"
  );

  if (whitespaceColumns.length > 0) {
    steps.push({
      id: "trim_whitespace",
      operation: "trim_whitespace",
      columns: whitespaceColumns,
      status: "proposed"
    });
  }

  booleanColumns.forEach((column) => {
    steps.push({
      id: `normalize_boolean:${column}`,
      operation: "normalize_boolean",
      column,
      trueValues: ["true", "yes", "1"],
      falseValues: ["false", "no", "0"],
      status: "proposed"
    });
  });

  if (missingTokenColumns.length > 0) {
    steps.push({
      id: "standardize_missing_tokens",
      operation: "standardize_missing_tokens",
      columns: missingTokenColumns,
      tokens: ["N/A", "NA", "null", "none", "-", "--", "?"],
      status: "proposed"
    });
  }

  return steps;
}

function buildDatasetSummary(file: File, rows: Record<string, unknown>[]) {
  const firstRow = rows[0];
  const fields = firstRow ? Object.keys(firstRow) : [];
  const warnings: string[] = [];

  if (fields.length === 0) {
    throw new Error("CSV has no usable header row.");
  }

  if (rows.length === 0) {
    throw new Error("CSV has no data rows.");
  }

  if (rows.length >= MAX_INFERENCE_ROWS) {
    warnings.push(
      `Schema was inferred from the first ${MAX_INFERENCE_ROWS.toLocaleString()} rows.`
    );
  }

  const columns = fields.map((field) => {
    const values = rows.map((row) => String(row[field] ?? ""));
    const presentValues = values.filter((value) => !isMissing(value));
    const uniqueValues = new Set(presentValues);
    const topValues = Array.from(
      presentValues.reduce((counts, value) => {
        counts.set(value, (counts.get(value) ?? 0) + 1);
        return counts;
      }, new Map<string, number>())
    )
      .map(([value, count]) => ({
        value,
        count,
        percent:
          presentValues.length === 0 ? 0 : (count / presentValues.length) * 100
      }))
      .sort((left, right) => {
        if (left.count !== right.count) return right.count - left.count;
        return left.value.localeCompare(right.value);
      })
      .slice(0, 5);
    const uniqueSamples = Array.from(new Set(presentValues.slice(0, 10))).slice(
      0,
      5
    );

    return {
      name: field,
      inferredType: inferColumnType(values),
      missingCount: values.filter(isMissing).length,
      nonMissingCount: presentValues.length,
      uniqueCount: uniqueValues.size,
      uniqueRatio:
        presentValues.length === 0
          ? 0
          : uniqueValues.size / presentValues.length,
      topValues,
      sampleValues: uniqueSamples,
      profilingNotes: buildProfilingNotes(field, values)
    };
  });

  return {
    fileName: file.name,
    fileSizeBytes: file.size,
    parsedRowCount: rows.length,
    columns,
    previewRows: rows
      .slice(0, MAX_PREVIEW_ROWS)
      .map((row) =>
        Object.fromEntries(
          fields.map((field) => [field, normalizePreviewValue(row[field])])
        )
      ),
    warnings,
    proposedPreprocessingSteps: buildPreprocessingSteps(columns)
  } satisfies DatasetSummary;
}

export function parseCsvFile(file: File): Promise<DatasetSummary> {
  return new Promise((resolve, reject) => {
    let generatedHeaderCount = 0;

    Papa.parse<Record<string, unknown>>(file, {
      header: true,
      preview: MAX_INFERENCE_ROWS,
      skipEmptyLines: true,
      transformHeader: (header, index) => {
        const trimmed = header.trim();
        if (trimmed) return trimmed;
        generatedHeaderCount += 1;
        return `unnamed_${index + 1}`;
      },
      complete: (results) => {
        const renamedHeaders = (
          results.meta as Papa.ParseMeta & {
            renamedHeaders?: Record<string, string>;
          }
        ).renamedHeaders;

        if (renamedHeaders && Object.keys(renamedHeaders).length > 0) {
          reject(new Error("CSV contains duplicate column names."));
          return;
        }

        const severeError = results.errors.find(
          (error) =>
            error.code !== "TooFewFields" && error.code !== "TooManyFields"
        );

        if (severeError) {
          reject(new Error(severeError.message || "CSV parsing failed."));
          return;
        }

        try {
          const summary = buildDatasetSummary(file, results.data);
          if (generatedHeaderCount > 0) {
            summary.warnings.push(
              `${generatedHeaderCount} empty column header${generatedHeaderCount === 1 ? " was" : "s were"} renamed to unnamed columns.`
            );
          }
          const fieldErrors = results.errors.filter(
            (error) =>
              error.code === "TooFewFields" || error.code === "TooManyFields"
          );
          if (fieldErrors.length > 0) {
            summary.warnings.push(
              `${fieldErrors.length} row${fieldErrors.length === 1 ? "" : "s"} had an inconsistent field count.`
            );
          }
          console.info("CSV upload parsed", {
            fileName: summary.fileName,
            fileSizeBytes: summary.fileSizeBytes,
            parsedRowCount: summary.parsedRowCount,
            columns: summary.columns.length
          });
          resolve(summary);
        } catch (error) {
          reject(error);
        }
      },
      error: (error) => reject(new Error(error.message))
    });
  });
}
