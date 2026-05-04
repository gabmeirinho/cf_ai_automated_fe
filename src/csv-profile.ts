import Papa from "papaparse";
import {
  evaluateFeatureExpression,
  getInputColumns,
  type ValidatedFeatureSuggestion
} from "./feature-engineering";

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

export type PreprocessingSuggestionAction = string;

export interface SelectedPreprocessingStep {
  columnName: string;
  action: PreprocessingSuggestionAction;
}

export interface PreparedFeatureColumn {
  name: string;
  sourceColumn: string;
  inferredType: InferredColumnType;
  preprocessingActions: PreprocessingSuggestionAction[];
  role: "passthrough" | "one_hot" | "split_name";
}

export interface PreparedFeatureContext {
  columns: PreparedFeatureColumn[];
  numericColumns: string[];
  dateColumns: string[];
  removedColumns: string[];
  preprocessingSteps: SelectedPreprocessingStep[];
  previewRows: Record<string, PreviewValue>[];
}

export interface CsvTransformationPlan {
  targetColumn: string;
  featureColumns: string[];
  preprocessingSteps: SelectedPreprocessingStep[];
  engineeredFeatures?: ValidatedFeatureSuggestion[];
}

export interface CsvTransformationResult {
  csv: string;
  rowCount: number;
  outputColumns: string[];
  audit: string[];
}

export interface SplitConfig {
  trainRatio: number;
  seed: number;
}

export interface SplitExportResult {
  trainCsv: string;
  testCsv: string;
  trainRowCount: number;
  testRowCount: number;
  outputColumns: string[];
  audit: string[];
}

export interface PreprocessingSuggestion {
  columnName: string;
  action: PreprocessingSuggestionAction;
  reason: string;
  implementation?: string;
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

function inferColumnTypeFromStats(column: ColumnProfiler): InferredColumnType {
  if (column.nonMissingCount === 0) return "empty";
  if (column.numberCount === column.nonMissingCount) return "number";
  if (column.booleanCount === column.nonMissingCount) return "boolean";
  if (column.dateCount === column.nonMissingCount) return "date";
  if (
    column.numberCount > 0 ||
    column.booleanCount > 0 ||
    column.dateCount > 0
  ) {
    return "mixed";
  }
  return "string";
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

export function isTrainingRow(
  rowIndex: number,
  seed: number,
  trainRatio: number
): boolean {
  const hash = (Math.imul(rowIndex + 1, 2654435761) ^ seed) >>> 0;
  return (hash % 10000) / 10000 < trainRatio;
}

function normalizeActionName(action: string) {
  return action.trim().toLowerCase().replaceAll("_", " ");
}

function isNoStepAction(action: string) {
  const normalized = normalizeActionName(action);
  return normalized === "none" || normalized === "no step";
}

function isDropColumnAction(action: string) {
  const normalized = normalizeActionName(action);
  return normalized === "drop" || normalized === "drop column";
}

function isTrimWhitespaceAction(action: string) {
  return /trim/.test(normalizeActionName(action));
}

function isLowercaseAction(action: string) {
  const normalized = normalizeActionName(action);
  return /lowercase|lower case|case fold/.test(normalized);
}

function isStandardizeMissingAction(action: string) {
  const normalized = normalizeActionName(action);
  return /standardize missing|missing value tokens|missing-value tokens/.test(
    normalized
  );
}

function isNormalizeBooleanAction(action: string) {
  const normalized = normalizeActionName(action);
  return /normalize boolean|standardize boolean/.test(normalized);
}

function isFillMeanAction(action: string) {
  return /fill missing.*mean|mean imputation/.test(normalizeActionName(action));
}

function isFillMedianAction(action: string) {
  return /fill missing.*median|median imputation/.test(
    normalizeActionName(action)
  );
}

function isFillModeAction(action: string) {
  const normalized = normalizeActionName(action);
  return /fill missing.*most common|fill missing.*mode|mode imputation/.test(
    normalized
  );
}

function isOneHotAction(action: string) {
  return /one hot|one-hot/.test(normalizeActionName(action));
}

function isSplitNameAction(action: string) {
  const normalized = normalizeActionName(action);
  return /split.*name|name.*parts|extract.*name/.test(normalized);
}

function isSupportedTransformationAction(action: string) {
  return (
    isNoStepAction(action) ||
    isDropColumnAction(action) ||
    isTrimWhitespaceAction(action) ||
    isLowercaseAction(action) ||
    isStandardizeMissingAction(action) ||
    isNormalizeBooleanAction(action) ||
    isFillMeanAction(action) ||
    isFillMedianAction(action) ||
    isFillModeAction(action) ||
    isOneHotAction(action) ||
    isSplitNameAction(action)
  );
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
      message: `${field} is missing at least ${Math.round(HIGH_MISSINGNESS_THRESHOLD * 100)}% of parsed rows.`,
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

interface ColumnProfiler {
  name: string;
  rowCount: number;
  missingCount: number;
  nonMissingCount: number;
  numberCount: number;
  booleanCount: number;
  dateCount: number;
  whitespaceCount: number;
  numericAfterTrimCount: number;
  booleanAfterCaseFoldCount: number;
  dateAfterTrimCount: number;
  missingLikeTokenCount: number;
  valueCounts: Map<string, number>;
  sampleValues: string[];
  caseCountsByTrimmedValue: Map<string, number>;
  caseVariantsByLowerValue: Map<string, Set<string>>;
}

interface CsvProfiler {
  fields: string[];
  columns: Map<string, ColumnProfiler>;
  previewRows: Record<string, PreviewValue>[];
  rowCount: number;
  fieldErrorCount: number;
}

function createColumnProfiler(name: string): ColumnProfiler {
  return {
    name,
    rowCount: 0,
    missingCount: 0,
    nonMissingCount: 0,
    numberCount: 0,
    booleanCount: 0,
    dateCount: 0,
    whitespaceCount: 0,
    numericAfterTrimCount: 0,
    booleanAfterCaseFoldCount: 0,
    dateAfterTrimCount: 0,
    missingLikeTokenCount: 0,
    valueCounts: new Map(),
    sampleValues: [],
    caseCountsByTrimmedValue: new Map(),
    caseVariantsByLowerValue: new Map()
  };
}

function createCsvProfiler(fields: string[]): CsvProfiler {
  return {
    fields,
    columns: new Map(
      fields.map((field) => [field, createColumnProfiler(field)])
    ),
    previewRows: [],
    rowCount: 0,
    fieldErrorCount: 0
  };
}

function addColumnValue(column: ColumnProfiler, rawValue: unknown) {
  const value = String(rawValue ?? "");
  const trimmed = value.trim();

  column.rowCount += 1;

  if (isMissing(value)) {
    column.missingCount += 1;
  } else {
    column.nonMissingCount += 1;
    column.valueCounts.set(value, (column.valueCounts.get(value) ?? 0) + 1);

    if (
      !column.sampleValues.includes(value) &&
      column.sampleValues.length < 5
    ) {
      column.sampleValues.push(value);
    }
  }

  if (looksNumber(value)) column.numberCount += 1;
  if (looksBoolean(value)) column.booleanCount += 1;
  if (looksDate(value)) column.dateCount += 1;

  if (value.length > 0 && value !== trimmed) {
    column.whitespaceCount += 1;
  }
  if (value !== trimmed && looksNumber(trimmed)) {
    column.numericAfterTrimCount += 1;
  }
  if (looksBoolean(trimmed) && trimmed !== trimmed.toLowerCase()) {
    column.booleanAfterCaseFoldCount += 1;
  }
  if (value !== trimmed && looksDate(trimmed)) {
    column.dateAfterTrimCount += 1;
  }
  if (isMissingLikeToken(value)) {
    column.missingLikeTokenCount += 1;
  }
  if (trimmed) {
    column.caseCountsByTrimmedValue.set(
      trimmed,
      (column.caseCountsByTrimmedValue.get(trimmed) ?? 0) + 1
    );
    const lower = trimmed.toLowerCase();
    const variants = column.caseVariantsByLowerValue.get(lower) ?? new Set();
    variants.add(trimmed);
    column.caseVariantsByLowerValue.set(lower, variants);
  }
}

function addProfilerRow(profiler: CsvProfiler, row: Record<string, unknown>) {
  profiler.rowCount += 1;

  if (profiler.previewRows.length < MAX_PREVIEW_ROWS) {
    profiler.previewRows.push(
      Object.fromEntries(
        profiler.fields.map((field) => [
          field,
          normalizePreviewValue(row[field])
        ])
      )
    );
  }

  profiler.fields.forEach((field) => {
    const column = profiler.columns.get(field);
    if (column) addColumnValue(column, row[field]);
  });
}

function getCaseVariantCount(column: ColumnProfiler) {
  let count = 0;

  for (const variants of column.caseVariantsByLowerValue.values()) {
    if (variants.size <= 1) continue;
    for (const variant of variants) {
      count += column.caseCountsByTrimmedValue.get(variant) ?? 0;
    }
  }

  return count;
}

function buildProfilingNotesFromProfiler(column: ColumnProfiler) {
  const notes: ProfilingNote[] = [];
  const uniqueCount = column.valueCounts.size;
  const missingRatio =
    column.rowCount === 0 ? 0 : column.missingCount / column.rowCount;
  const caseVariantCount = getCaseVariantCount(column);

  if (column.whitespaceCount > 0) {
    notes.push({
      code: "leading_trailing_whitespace",
      message: `${column.name} contains leading or trailing whitespace. Raw preview keeps those values unchanged.`,
      affectedCount: column.whitespaceCount
    });
  }
  if (caseVariantCount > 0) {
    notes.push({
      code: "case_variants",
      message: `${column.name} contains values that differ only by letter case.`,
      affectedCount: caseVariantCount
    });
  }
  if (column.numericAfterTrimCount > 0) {
    notes.push({
      code: "numeric_after_trim",
      message: `${column.name} has numeric-like values after trimming whitespace.`,
      affectedCount: column.numericAfterTrimCount
    });
  }
  if (column.booleanAfterCaseFoldCount > 0) {
    notes.push({
      code: "boolean_after_case_fold",
      message: `${column.name} has boolean-like values after case normalization.`,
      affectedCount: column.booleanAfterCaseFoldCount
    });
  }
  if (column.dateAfterTrimCount > 0) {
    notes.push({
      code: "date_after_trim",
      message: `${column.name} has date-like values after trimming whitespace.`,
      affectedCount: column.dateAfterTrimCount
    });
  }
  if (column.missingLikeTokenCount > 0) {
    notes.push({
      code: "missing_like_tokens",
      message: `${column.name} contains tokens that commonly represent missing values.`,
      affectedCount: column.missingLikeTokenCount
    });
  }
  if (column.missingCount === column.rowCount) {
    notes.push({
      code: "empty_column",
      message: `${column.name} has no non-empty values.`,
      affectedCount: column.missingCount
    });
  } else if (missingRatio >= HIGH_MISSINGNESS_THRESHOLD) {
    notes.push({
      code: "high_missingness",
      message: `${column.name} is missing at least ${Math.round(HIGH_MISSINGNESS_THRESHOLD * 100)}% of parsed rows.`,
      affectedCount: column.missingCount
    });
  }
  if (uniqueCount === 1 && column.nonMissingCount > 0) {
    notes.push({
      code: "constant_column",
      message: `${column.name} has only one unique non-empty value.`,
      affectedCount: column.nonMissingCount
    });
  }
  if (
    column.nonMissingCount >= MIN_IDENTIFIER_ROW_COUNT &&
    uniqueCount / column.rowCount >= LIKELY_IDENTIFIER_UNIQUE_RATIO
  ) {
    notes.push({
      code: "likely_identifier",
      message: `${column.name} is likely an identifier because most non-empty values are unique.`,
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

function buildDatasetSummary(file: File, profiler: CsvProfiler) {
  const fields = profiler.fields;
  const warnings: string[] = [];

  if (fields.length === 0) {
    throw new Error("CSV has no usable header row.");
  }

  if (profiler.rowCount === 0) {
    throw new Error("CSV has no data rows.");
  }

  const columns = fields.map((field) => {
    const column = profiler.columns.get(field) ?? createColumnProfiler(field);
    const topValues = Array.from(column.valueCounts)
      .map(([value, count]) => ({
        value,
        count,
        percent:
          column.nonMissingCount === 0
            ? 0
            : (count / column.nonMissingCount) * 100
      }))
      .sort((left, right) => {
        if (left.count !== right.count) return right.count - left.count;
        return left.value.localeCompare(right.value);
      })
      .slice(0, 5);

    return {
      name: field,
      inferredType: inferColumnTypeFromStats(column),
      missingCount: column.missingCount,
      nonMissingCount: column.nonMissingCount,
      uniqueCount: column.valueCounts.size,
      uniqueRatio:
        column.nonMissingCount === 0
          ? 0
          : column.valueCounts.size / column.nonMissingCount,
      topValues,
      sampleValues: column.sampleValues,
      profilingNotes: buildProfilingNotesFromProfiler(column)
    };
  });

  return {
    fileName: file.name,
    fileSizeBytes: file.size,
    parsedRowCount: profiler.rowCount,
    columns,
    previewRows: profiler.previewRows,
    warnings,
    proposedPreprocessingSteps: buildPreprocessingSteps(columns)
  } satisfies DatasetSummary;
}

function sanitizeColumnSuffix(value: string) {
  const suffix = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 48);
  return suffix || "value";
}

function getCell(row: Record<string, unknown>, columnName: string) {
  return String(row[columnName] ?? "");
}

function median(values: number[]) {
  if (values.length === 0) return "";
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return String(sorted[midpoint]);
  return String((sorted[midpoint - 1] + sorted[midpoint]) / 2);
}

function mode(values: string[]) {
  const counts = new Map<string, number>();
  values
    .filter((value) => !isMissing(value))
    .forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1));

  const first = Array.from(counts).sort((left, right) => {
    if (left[1] !== right[1]) return right[1] - left[1];
    return left[0].localeCompare(right[0]);
  })[0];

  return first?.[0] ?? "";
}

function parseFullName(value: string) {
  const withoutTitle = value.replace(
    /^[^,]+,\s*(Mr|Mrs|Miss|Ms|Master|Dr|Rev|Col|Major|Sir|Lady|Don|Dona)\.?\s+/i,
    ""
  );
  const normalized = withoutTitle.includes(",")
    ? withoutTitle.split(",").reverse().join(" ")
    : withoutTitle;
  const parts = normalized.trim().split(/\s+/).filter(Boolean);

  return {
    first: parts[0] ?? "",
    last: parts.length > 1 ? parts[parts.length - 1] : ""
  };
}

function groupStepsByColumn(steps: SelectedPreprocessingStep[]) {
  const grouped = new Map<string, string[]>();

  steps.forEach((step) => {
    if (isNoStepAction(step.action)) return;
    const current = grouped.get(step.columnName) ?? [];
    if (!current.includes(step.action)) current.push(step.action);
    grouped.set(step.columnName, current);
  });

  return grouped;
}

function valuesForPreparedCategories(
  profile: AiReviewProfile,
  columnName: string
) {
  const column = profile.columns.find((item) => item.name === columnName);
  const values = [
    ...(column?.topValues.map((item) => item.value) ?? []),
    ...(column?.sampleValues ?? []),
    ...profile.previewRows.map((row) => row[columnName])
  ];

  return Array.from(
    new Set(
      values
        .map((value) => (value == null ? "" : String(value).trim()))
        .filter(Boolean)
    )
  ).sort((left, right) => left.localeCompare(right));
}

export function buildPreparedFeatureContext(
  profile: AiReviewProfile,
  keptColumns: string[],
  preprocessingSteps: SelectedPreprocessingStep[]
): PreparedFeatureContext {
  const stepsByColumn = groupStepsByColumn(preprocessingSteps);
  const columnsByName = new Map(
    profile.columns.map((column) => [column.name, column])
  );
  const preparedColumns: PreparedFeatureColumn[] = [];
  const removedColumns: string[] = [];
  const previewRows: Record<string, PreviewValue>[] = profile.previewRows.map(
    () => ({})
  );

  keptColumns.forEach((columnName) => {
    const column = columnsByName.get(columnName);
    if (!column) return;

    const actions = sortColumnActions(stepsByColumn.get(columnName) ?? []);

    if (actions.some(isDropColumnAction)) {
      removedColumns.push(columnName);
      return;
    }

    if (actions.some(isSplitNameAction)) {
      const firstName = `${columnName}_first`;
      const lastName = `${columnName}_last`;
      preparedColumns.push(
        {
          name: firstName,
          sourceColumn: columnName,
          inferredType: "string",
          preprocessingActions: actions,
          role: "split_name"
        },
        {
          name: lastName,
          sourceColumn: columnName,
          inferredType: "string",
          preprocessingActions: actions,
          role: "split_name"
        }
      );
      profile.previewRows.forEach((row, rowIndex) => {
        const transformedValue = transformColumnValue(
          row[columnName] == null ? "" : String(row[columnName]),
          actions,
          {}
        );
        const parsedName = parseFullName(transformedValue);
        previewRows[rowIndex][firstName] = parsedName.first;
        previewRows[rowIndex][lastName] = parsedName.last;
      });
      return;
    }

    if (actions.some(isOneHotAction)) {
      const categories = valuesForPreparedCategories(profile, columnName);
      categories.forEach((category) => {
        const preparedName = `${columnName}__${sanitizeColumnSuffix(category)}`;
        preparedColumns.push({
          name: preparedName,
          sourceColumn: columnName,
          inferredType: "number",
          preprocessingActions: actions,
          role: "one_hot"
        });
        profile.previewRows.forEach((row, rowIndex) => {
          const transformedValue = transformColumnValue(
            row[columnName] == null ? "" : String(row[columnName]),
            actions,
            {}
          );
          previewRows[rowIndex][preparedName] =
            transformedValue.trim() === category ? 1 : 0;
        });
      });
      return;
    }

    preparedColumns.push({
      name: columnName,
      sourceColumn: columnName,
      inferredType: column.inferredType,
      preprocessingActions: actions,
      role: "passthrough"
    });
    profile.previewRows.forEach((row, rowIndex) => {
      previewRows[rowIndex][columnName] = transformColumnValue(
        row[columnName] == null ? "" : String(row[columnName]),
        actions,
        {}
      );
    });
  });

  return {
    columns: preparedColumns,
    numericColumns: preparedColumns
      .filter((column) => column.inferredType === "number")
      .map((column) => column.name),
    dateColumns: preparedColumns
      .filter((column) => column.inferredType === "date")
      .map((column) => column.name),
    removedColumns,
    preprocessingSteps,
    previewRows
  };
}

function sortColumnActions(actions: string[]) {
  const priority = (action: string) => {
    if (isStandardizeMissingAction(action)) return 10;
    if (isTrimWhitespaceAction(action)) return 20;
    if (isLowercaseAction(action)) return 30;
    if (isNormalizeBooleanAction(action)) return 40;
    if (isFillMeanAction(action) || isFillMedianAction(action)) return 50;
    if (isFillModeAction(action)) return 60;
    if (isSplitNameAction(action)) return 70;
    if (isOneHotAction(action)) return 80;
    if (isDropColumnAction(action)) return 90;
    return 100;
  };

  return [...actions].sort((left, right) => priority(left) - priority(right));
}

function buildTransformStats(
  rows: Record<string, unknown>[],
  featureColumns: string[],
  stepsByColumn: Map<string, string[]>
) {
  const stats = new Map<
    string,
    {
      mean?: string;
      median?: string;
      mode?: string;
      categories?: string[];
    }
  >();

  featureColumns.forEach((columnName) => {
    const actions = stepsByColumn.get(columnName) ?? [];
    const values = rows.map((row) => getCell(row, columnName));
    const numericValues = values
      .filter((value) => !isMissing(value) && looksNumber(value))
      .map(Number);
    const categories = Array.from(
      new Set(values.map((value) => value.trim()).filter(Boolean))
    ).sort((left, right) => left.localeCompare(right));
    const columnStats: {
      mean?: string;
      median?: string;
      mode?: string;
      categories?: string[];
    } = {};

    if (actions.some(isFillMeanAction) && numericValues.length > 0) {
      columnStats.mean = String(
        numericValues.reduce((sum, value) => sum + value, 0) /
          numericValues.length
      );
    }
    if (actions.some(isFillMedianAction)) {
      columnStats.median = median(numericValues);
    }
    if (actions.some(isFillModeAction)) {
      columnStats.mode = mode(values);
    }
    if (actions.some(isOneHotAction)) {
      columnStats.categories = categories;
    }

    stats.set(columnName, columnStats);
  });

  return stats;
}

function transformColumnValue(
  value: string,
  actions: string[],
  columnStats: {
    mean?: string;
    median?: string;
    mode?: string;
  }
) {
  let nextValue = value;

  sortColumnActions(actions).forEach((action) => {
    if (isStandardizeMissingAction(action) && isMissingLikeToken(nextValue)) {
      nextValue = "";
    } else if (isTrimWhitespaceAction(action)) {
      nextValue = nextValue.trim();
    } else if (isLowercaseAction(action)) {
      nextValue = nextValue.toLowerCase();
    } else if (isNormalizeBooleanAction(action) && looksBoolean(nextValue)) {
      nextValue = /^(true|1|yes)$/i.test(nextValue.trim()) ? "true" : "false";
    } else if (isFillMeanAction(action) && isMissing(nextValue)) {
      nextValue = columnStats.mean ?? nextValue;
    } else if (isFillMedianAction(action) && isMissing(nextValue)) {
      nextValue = columnStats.median ?? nextValue;
    } else if (isFillModeAction(action) && isMissing(nextValue)) {
      nextValue = columnStats.mode ?? nextValue;
    }
  });

  return nextValue;
}

function validateTransformationPlan(
  fields: string[],
  plan: CsvTransformationPlan,
  preparedFeatureColumns: string[]
) {
  const fieldSet = new Set(fields);
  const featureSet = new Set(plan.featureColumns);
  const preparedFeatureSet = new Set(preparedFeatureColumns);
  const generatedFeatureNames = new Set<string>();

  if (!fieldSet.has(plan.targetColumn)) {
    throw new Error("The target column is not present in the CSV.");
  }
  if (featureSet.has(plan.targetColumn)) {
    throw new Error("Target leakage blocked: target cannot be a feature.");
  }

  plan.featureColumns.forEach((columnName) => {
    if (!fieldSet.has(columnName)) {
      throw new Error(`${columnName} is not present in the CSV.`);
    }
  });

  plan.preprocessingSteps.forEach((step) => {
    if (step.columnName === plan.targetColumn) {
      throw new Error("Target leakage blocked: target cannot be transformed.");
    }
    if (!featureSet.has(step.columnName)) {
      throw new Error(
        `${step.columnName} cannot be transformed because it is not a kept feature.`
      );
    }
    if (!isSupportedTransformationAction(step.action)) {
      throw new Error(`${step.action} is not a supported export transform.`);
    }
  });

  (plan.engineeredFeatures ?? []).forEach((feature) => {
    if (fieldSet.has(feature.name) || preparedFeatureSet.has(feature.name)) {
      throw new Error(
        `${feature.name} cannot be generated because it already exists in the prepared dataset.`
      );
    }
    if (generatedFeatureNames.has(feature.name)) {
      throw new Error(
        `${feature.name} cannot be generated more than once in the same export.`
      );
    }
    generatedFeatureNames.add(feature.name);

    getInputColumns(feature.expression).forEach((columnName) => {
      if (columnName === plan.targetColumn) {
        throw new Error(
          "Target leakage blocked: engineered features cannot use the target."
        );
      }
      if (!preparedFeatureSet.has(columnName)) {
        throw new Error(
          `${feature.name} references ${columnName}, which is not present after preprocessing.`
        );
      }
    });
  });
}

function getPreparedFeatureColumnNames(
  plan: CsvTransformationPlan,
  stats: Map<
    string,
    { mean?: string; median?: string; mode?: string; categories?: string[] }
  >
) {
  return plan.featureColumns.flatMap((columnName) => {
    const actions = sortColumnActions(
      plan.preprocessingSteps
        .filter((step) => step.columnName === columnName)
        .map((step) => step.action)
    );

    if (actions.some(isDropColumnAction)) return [];
    if (actions.some(isSplitNameAction)) {
      return [`${columnName}_first`, `${columnName}_last`];
    }
    if (actions.some(isOneHotAction)) {
      return (stats.get(columnName)?.categories ?? []).map(
        (category) => `${columnName}__${sanitizeColumnSuffix(category)}`
      );
    }
    return [columnName];
  });
}

function formatEngineeredFeatureValue(value: number | null) {
  return value === null ? "" : String(value);
}

function transformRow(
  row: Record<string, unknown>,
  plan: CsvTransformationPlan,
  stepsByColumn: Map<string, string[]>,
  stats: Map<
    string,
    { mean?: string; median?: string; mode?: string; categories?: string[] }
  >
) {
  const output: Record<string, string> = {};

  plan.featureColumns.forEach((columnName) => {
    const actions = sortColumnActions(stepsByColumn.get(columnName) ?? []);
    const columnStats = stats.get(columnName) ?? {};
    const transformedValue = transformColumnValue(
      getCell(row, columnName),
      actions,
      columnStats
    );

    if (actions.some(isDropColumnAction)) return;

    if (actions.some(isSplitNameAction)) {
      const parsedName = parseFullName(transformedValue);
      output[`${columnName}_first`] = parsedName.first;
      output[`${columnName}_last`] = parsedName.last;
      return;
    }

    if (actions.some(isOneHotAction)) {
      const categories = columnStats.categories ?? [];
      categories.forEach((category) => {
        output[`${columnName}__${sanitizeColumnSuffix(category)}`] =
          transformedValue.trim() === category ? "1" : "0";
      });
      return;
    }

    output[columnName] = transformedValue;
  });

  (plan.engineeredFeatures ?? []).forEach((feature) => {
    output[feature.name] = formatEngineeredFeatureValue(
      evaluateFeatureExpression(feature.expression, output)
    );
  });

  output[plan.targetColumn] = getCell(row, plan.targetColumn);
  return output;
}

export async function transformCsvFile(
  file: File,
  plan: CsvTransformationPlan,
  splitConfig?: SplitConfig
): Promise<CsvTransformationResult | SplitExportResult> {
  const source = await file.text();
  const parsed = Papa.parse<Record<string, unknown>>(source, {
    header: true,
    skipEmptyLines: true,
    transformHeader: (header, index) => header.trim() || `unnamed_${index + 1}`
  });

  const duplicateHeaders = (
    parsed.meta as Papa.ParseMeta & {
      renamedHeaders?: Record<string, string>;
    }
  ).renamedHeaders;

  if (duplicateHeaders && Object.keys(duplicateHeaders).length > 0) {
    throw new Error("CSV contains duplicate column names.");
  }

  const severeError = parsed.errors.find(
    (error) => error.code !== "TooFewFields" && error.code !== "TooManyFields"
  );
  if (severeError) {
    throw new Error(severeError.message || "CSV parsing failed.");
  }

  const fields = parsed.meta.fields ?? Object.keys(parsed.data[0] ?? {});

  const stepsByColumn = groupStepsByColumn(plan.preprocessingSteps);

  if (splitConfig) {
    // --- Split path: fit on training, transform both ---
    const trainRows: typeof parsed.data = [];
    const testRows: typeof parsed.data = [];

    parsed.data.forEach((row, index) => {
      if (isTrainingRow(index, splitConfig.seed, splitConfig.trainRatio)) {
        trainRows.push(row);
      } else {
        testRows.push(row);
      }
    });

    // Fit statistics on training rows only
    const trainStats = buildTransformStats(
      trainRows,
      plan.featureColumns,
      stepsByColumn
    );
    validateTransformationPlan(
      fields,
      plan,
      getPreparedFeatureColumnNames(plan, trainStats)
    );

    // Transform both splits using the same fitted stats
    const trainOutputRows = trainRows.map((row) =>
      transformRow(row, plan, stepsByColumn, trainStats)
    );
    const testOutputRows = testRows.map((row) =>
      transformRow(row, plan, stepsByColumn, trainStats)
    );

    const outputColumns = Array.from(
      new Set(
        [...trainOutputRows, ...testOutputRows].flatMap((row) =>
          Object.keys(row)
        )
      )
    );

    // Strip one-hot columns from test that weren't seen in training
    const trainingCategoryColumns = new Set<string>();
    plan.featureColumns.forEach((columnName) => {
      const stats = trainStats.get(columnName);
      stats?.categories?.forEach((cat) =>
        trainingCategoryColumns.add(
          `${columnName}__${sanitizeColumnSuffix(cat)}`
        )
      );
    });

    const sanitizedTestRows = testOutputRows.map((row) => {
      const sanitized: Record<string, string> = {};
      Object.entries(row).forEach(([key, value]) => {
        if (
          key === plan.targetColumn ||
          !key.includes("__") ||
          trainingCategoryColumns.has(key)
        ) {
          sanitized[key] = value;
        }
      });
      return sanitized;
    });

    const audit = [
      `Target ${plan.targetColumn} was kept unchanged and excluded from feature transforms.`,
      `${plan.featureColumns.length.toLocaleString()} feature column${plan.featureColumns.length === 1 ? "" : "s"} were eligible for export.`,
      `${(plan.engineeredFeatures ?? []).length.toLocaleString()} accepted engineered feature${(plan.engineeredFeatures ?? []).length === 1 ? "" : "s"} added.`,
      `${plan.preprocessingSteps.filter((step) => !isNoStepAction(step.action)).length.toLocaleString()} preprocessing step${plan.preprocessingSteps.length === 1 ? "" : "s"} applied.`,
      `Fitted statistics computed from training set only.`,
      `${testRows.length.toLocaleString()} test rows held out during fit.`
    ];

    return {
      trainCsv: Papa.unparse(trainOutputRows, { columns: outputColumns }),
      testCsv: Papa.unparse(sanitizedTestRows, { columns: outputColumns }),
      trainRowCount: trainOutputRows.length,
      testRowCount: sanitizedTestRows.length,
      outputColumns,
      audit
    };
  }

  // --- No-split path (existing behaviour) ---
  const transformStats = buildTransformStats(
    parsed.data,
    plan.featureColumns,
    stepsByColumn
  );
  validateTransformationPlan(
    fields,
    plan,
    getPreparedFeatureColumnNames(plan, transformStats)
  );
  const outputRows = parsed.data.map((row) =>
    transformRow(row, plan, stepsByColumn, transformStats)
  );
  const outputColumns = Array.from(
    outputRows.reduce((columns, row) => {
      Object.keys(row).forEach((columnName) => columns.add(columnName));
      return columns;
    }, new Set<string>())
  );
  const audit = [
    `Target ${plan.targetColumn} was kept unchanged and excluded from feature transforms.`,
    `${plan.featureColumns.length.toLocaleString()} feature column${plan.featureColumns.length === 1 ? "" : "s"} were eligible for export.`,
    `${(plan.engineeredFeatures ?? []).length.toLocaleString()} accepted engineered feature${(plan.engineeredFeatures ?? []).length === 1 ? "" : "s"} added.`,
    `${plan.preprocessingSteps.filter((step) => !isNoStepAction(step.action)).length.toLocaleString()} preprocessing step${plan.preprocessingSteps.length === 1 ? "" : "s"} applied.`,
    "No transform reads from the target column."
  ];

  return {
    csv: Papa.unparse(outputRows, { columns: outputColumns }),
    rowCount: outputRows.length,
    outputColumns,
    audit
  };
}

export async function parseCsvFile(
  file: File,
  splitConfig?: SplitConfig
): Promise<DatasetSummary> {
  const source =
    !("FileReader" in globalThis) && !("FileReaderSync" in globalThis)
      ? await file.text()
      : file;

  return new Promise((resolve, reject) => {
    let generatedHeaderCount = 0;
    let profiler: CsvProfiler | null = null;
    let failed = false;
    let globalRowIndex = 0;

    const rejectOnce = (error: Error) => {
      if (failed) return;
      failed = true;
      reject(error);
    };

    Papa.parse<Record<string, unknown>>(source, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (header, index) => {
        const trimmed = header.trim();
        if (trimmed) return trimmed;
        generatedHeaderCount += 1;
        return `unnamed_${index + 1}`;
      },
      chunk: (results, parser) => {
        const renamedHeaders = (
          results.meta as Papa.ParseMeta & {
            renamedHeaders?: Record<string, string>;
          }
        ).renamedHeaders;

        if (renamedHeaders && Object.keys(renamedHeaders).length > 0) {
          rejectOnce(new Error("CSV contains duplicate column names."));
          parser.abort();
          return;
        }

        const severeError = results.errors.find(
          (error) =>
            error.code !== "TooFewFields" && error.code !== "TooManyFields"
        );

        if (severeError) {
          rejectOnce(new Error(severeError.message || "CSV parsing failed."));
          parser.abort();
          return;
        }

        if (!profiler) {
          const fields =
            results.meta.fields ?? Object.keys(results.data[0] ?? {});
          profiler = createCsvProfiler(fields);
        }

        profiler.fieldErrorCount += results.errors.filter(
          (error) =>
            error.code === "TooFewFields" || error.code === "TooManyFields"
        ).length;

        results.data.forEach((row) => {
          if (!profiler) return;
          if (
            splitConfig &&
            !isTrainingRow(
              globalRowIndex,
              splitConfig.seed,
              splitConfig.trainRatio
            )
          ) {
            globalRowIndex += 1;
            return;
          }
          addProfilerRow(profiler, row);
          globalRowIndex += 1;
        });
      },
      complete: () => {
        if (failed) return;

        const completedProfiler = profiler ?? createCsvProfiler([]);

        try {
          const summary = buildDatasetSummary(file, completedProfiler);
          if (completedProfiler.rowCount > MAX_INFERENCE_ROWS) {
            summary.warnings.push(
              `Profiled all ${completedProfiler.rowCount.toLocaleString()} parsed rows.`
            );
          }
          if (generatedHeaderCount > 0) {
            summary.warnings.push(
              `${generatedHeaderCount} empty column header${generatedHeaderCount === 1 ? " was" : "s were"} renamed to unnamed columns.`
            );
          }
          if (completedProfiler.fieldErrorCount > 0) {
            summary.warnings.push(
              `${completedProfiler.fieldErrorCount} row${completedProfiler.fieldErrorCount === 1 ? "" : "s"} had an inconsistent field count.`
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
          rejectOnce(error instanceof Error ? error : new Error(String(error)));
        }
      },
      error: (error) => rejectOnce(new Error(error.message))
    });
  });
}
