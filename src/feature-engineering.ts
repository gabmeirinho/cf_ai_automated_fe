export type FeatureExpression =
  | { op: "log1p"; column: string }
  | { op: "sqrt"; column: string }
  | { op: "square"; column: string }
  | { op: "ratio"; numerator: string; denominator: string }
  | { op: "difference"; left: string; right: string }
  | { op: "product"; left: string; right: string }
  | { op: "date_year"; column: string }
  | { op: "date_month"; column: string }
  | { op: "date_dayofweek"; column: string };

export interface AiFeatureSuggestion {
  expression: FeatureExpression;
  name?: string;
  reason: string;
  expectedBenefit: string;
  riskNotes?: string[];
}

export interface FeatureValidationContext {
  availableColumns: string[];
  targetColumn?: string;
  numericColumns: string[];
  dateColumns: string[];
  blockedColumns?: string[];
  existingColumns?: string[];
}

export interface ValidatedFeatureSuggestion {
  expression: FeatureExpression;
  name: string;
  reason: string;
  expectedBenefit: string;
  warnings: string[];
}

export interface RejectedFeatureSuggestion {
  suggestion: unknown;
  reason: string;
}

export interface FeatureValidationResult {
  accepted: ValidatedFeatureSuggestion[];
  rejected: RejectedFeatureSuggestion[];
}

export function validateFeatureSuggestions(
  suggestions: unknown[],
  context: FeatureValidationContext
): FeatureValidationResult {
  const accepted: ValidatedFeatureSuggestion[] = [];
  const rejected: RejectedFeatureSuggestion[] = [];

  const availableColumns = new Set(context.availableColumns);
  const numericColumns = new Set(context.numericColumns);
  const dateColumns = new Set(context.dateColumns);
  const blockedColumns = new Set(context.blockedColumns ?? []);
  const existingColumns = new Set(context.existingColumns ?? []);
  const generatedNames = new Set<string>();

  for (const suggestion of suggestions) {
    if (!isAiFeatureSuggestion(suggestion)) {
      rejected.push({
        suggestion,
        reason: "Suggestion does not match the expected AI feature suggestion shape."
      });
      continue;
    }

    const expression = suggestion.expression;

    if (!isFeatureExpression(expression)) {
      rejected.push({
        suggestion,
        reason: "Expression does not match the allowed feature grammar."
      });
      continue;
    }

    const inputColumns = getInputColumns(expression);

    const unknownColumn = inputColumns.find((column) => !availableColumns.has(column));
    if (unknownColumn) {
      rejected.push({
        suggestion,
        reason: `Expression references an unknown column: "${unknownColumn}".`
      });
      continue;
    }

    if (context.targetColumn && inputColumns.includes(context.targetColumn)) {
      rejected.push({
        suggestion,
        reason: "Expression uses the target column, which could create leakage."
      });
      continue;
    }

    const blockedColumn = inputColumns.find((column) => blockedColumns.has(column));
    if (blockedColumn) {
      rejected.push({
        suggestion,
        reason: `Expression uses a blocked column: "${blockedColumn}".`
      });
      continue;
    }

    if (!isCompatibleWithColumnTypes(expression, numericColumns, dateColumns)) {
      rejected.push({
        suggestion,
        reason: "Expression is not compatible with the inferred column types."
      });
      continue;
    }

    const proposedName = suggestion.name?.trim() || getFeatureName(expression);
    const safeName = makeSafeFeatureName(proposedName);

    if (!safeName) {
      rejected.push({
        suggestion,
        reason: "Generated feature name is empty after sanitization."
      });
      continue;
    }

    if (existingColumns.has(safeName)) {
      rejected.push({
        suggestion,
        reason: `Generated feature name "${safeName}" already exists in the dataset.`
      });
      continue;
    }

    if (generatedNames.has(safeName)) {
      rejected.push({
        suggestion,
        reason: `Generated feature name "${safeName}" is duplicated.`
      });
      continue;
    }

    generatedNames.add(safeName);

    accepted.push({
      expression,
      name: safeName,
      reason: suggestion.reason,
      expectedBenefit: suggestion.expectedBenefit,
      warnings: suggestion.riskNotes ?? []
    });
  }

  return { accepted, rejected };
}

export function applyFeatureEngineering(
  data: Record<string, unknown>[],
  expressions: FeatureExpression[]
): Record<string, unknown>[] {
  return data.map((row) => {
    const newRow: Record<string, unknown> = { ...row };

    for (const expr of expressions) {
      const featureName = getFeatureName(expr);
      newRow[featureName] = evaluateFeatureExpression(expr, row);
    }

    return newRow;
  });
}

export function applyValidatedFeatureSuggestions(
  data: Record<string, unknown>[],
  suggestions: ValidatedFeatureSuggestion[]
): Record<string, unknown>[] {
  return data.map((row) => {
    const newRow: Record<string, unknown> = { ...row };

    for (const suggestion of suggestions) {
      newRow[suggestion.name] = evaluateFeatureExpression(
        suggestion.expression,
        row
      );
    }

    return newRow;
  });
}

export function evaluateFeatureExpression(
  expr: FeatureExpression,
  row: Record<string, unknown>
): number | null {
  switch (expr.op) {
    case "log1p": {
      const value = toNumber(row[expr.column]);
      if (value === null || value < -1) return null;
      return Math.log1p(value);
    }

    case "sqrt": {
      const value = toNumber(row[expr.column]);
      if (value === null || value < 0) return null;
      return Math.sqrt(value);
    }

    case "square": {
      const value = toNumber(row[expr.column]);
      if (value === null) return null;
      return value ** 2;
    }

    case "ratio": {
      const numerator = toNumber(row[expr.numerator]);
      const denominator = toNumber(row[expr.denominator]);

      if (numerator === null || denominator === null || denominator === 0) {
        return null;
      }

      return numerator / denominator;
    }

    case "difference": {
      const left = toNumber(row[expr.left]);
      const right = toNumber(row[expr.right]);

      if (left === null || right === null) return null;

      return left - right;
    }

    case "product": {
      const left = toNumber(row[expr.left]);
      const right = toNumber(row[expr.right]);

      if (left === null || right === null) return null;

      return left * right;
    }

    case "date_year": {
      const date = toDate(row[expr.column]);
      if (date === null) return null;
      return date.getFullYear();
    }

    case "date_month": {
      const date = toDate(row[expr.column]);
      if (date === null) return null;
      return date.getMonth() + 1;
    }

    case "date_dayofweek": {
      const date = toDate(row[expr.column]);
      if (date === null) return null;
      return date.getDay();
    }
  }
}

export function getFeatureName(expr: FeatureExpression): string {
  switch (expr.op) {
    case "log1p":
      return makeSafeFeatureName(`fe_${expr.column}_log1p`);

    case "sqrt":
      return makeSafeFeatureName(`fe_${expr.column}_sqrt`);

    case "square":
      return makeSafeFeatureName(`fe_${expr.column}_square`);

    case "ratio":
      return makeSafeFeatureName(`fe_${expr.numerator}_to_${expr.denominator}`);

    case "difference":
      return makeSafeFeatureName(`fe_${expr.left}_minus_${expr.right}`);

    case "product":
      return makeSafeFeatureName(`fe_${expr.left}_times_${expr.right}`);

    case "date_year":
      return makeSafeFeatureName(`fe_${expr.column}_year`);

    case "date_month":
      return makeSafeFeatureName(`fe_${expr.column}_month`);

    case "date_dayofweek":
      return makeSafeFeatureName(`fe_${expr.column}_dayofweek`);
  }
}

export function getInputColumns(expr: FeatureExpression): string[] {
  switch (expr.op) {
    case "log1p":
    case "sqrt":
    case "square":
    case "date_year":
    case "date_month":
    case "date_dayofweek":
      return [expr.column];

    case "ratio":
      return [expr.numerator, expr.denominator];

    case "difference":
    case "product":
      return [expr.left, expr.right];
  }
}

export function isFeatureExpression(value: unknown): value is FeatureExpression {
  if (!isRecord(value)) return false;

  switch (value.op) {
    case "log1p":
    case "sqrt":
    case "square":
    case "date_year":
    case "date_month":
    case "date_dayofweek":
      return typeof value.column === "string" && value.column.trim() !== "";

    case "ratio":
      return (
        typeof value.numerator === "string" &&
        value.numerator.trim() !== "" &&
        typeof value.denominator === "string" &&
        value.denominator.trim() !== ""
      );

    case "difference":
    case "product":
      return (
        typeof value.left === "string" &&
        value.left.trim() !== "" &&
        typeof value.right === "string" &&
        value.right.trim() !== ""
      );

    default:
      return false;
  }
}

export function isAiFeatureSuggestion(
  value: unknown
): value is AiFeatureSuggestion {
  if (!isRecord(value)) return false;

  if (!isFeatureExpression(value.expression)) return false;
  if (typeof value.reason !== "string" || value.reason.trim() === "") return false;

  if (
    typeof value.expectedBenefit !== "string" ||
    value.expectedBenefit.trim() === ""
  ) {
    return false;
  }

  if (value.name !== undefined && typeof value.name !== "string") {
    return false;
  }

  if (value.riskNotes !== undefined) {
    if (!Array.isArray(value.riskNotes)) return false;
    if (!value.riskNotes.every((note) => typeof note === "string")) return false;
  }

  return true;
}

export function makeSafeFeatureName(name: string): string {
  const safe = name
    .trim()
    .replace(/[^a-zA-Z0-9_]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");

  if (safe === "") return "";

  if (safe.startsWith("fe_")) return safe;

  return `fe_${safe}`;
}

function isCompatibleWithColumnTypes(
  expr: FeatureExpression,
  numericColumns: Set<string>,
  dateColumns: Set<string>
): boolean {
  switch (expr.op) {
    case "log1p":
    case "sqrt":
    case "square":
      return numericColumns.has(expr.column);

    case "ratio":
      return (
        numericColumns.has(expr.numerator) &&
        numericColumns.has(expr.denominator)
      );

    case "difference":
    case "product":
      return numericColumns.has(expr.left) && numericColumns.has(expr.right);

    case "date_year":
    case "date_month":
    case "date_dayofweek":
      return dateColumns.has(expr.column);
  }
}

function toNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }

  const text = String(value).trim();

  if (text === "") return null;

  const normalized = text.replace(",", ".");
  const number = Number(normalized);

  return Number.isFinite(number) ? number : null;
}

function toDate(value: unknown): Date | null {
  if (value === null || value === undefined) return null;

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  const text = String(value).trim();

  if (text === "") return null;

  const date = new Date(text);

  if (Number.isNaN(date.getTime())) return null;

  return date;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}