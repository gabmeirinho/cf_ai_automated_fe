import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import {
  convertToModelMessages,
  generateText,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage
} from "ai";
import { createWorkersAI } from "workers-ai-provider";
import { z } from "zod";

const DEFAULT_AI_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8";
const JSON_GENERATION_SETTINGS = {
  maxOutputTokens: 4096,
  temperature: 0
} as const;

const columnAssumptionSchema = z.object({
  id: z.string(),
  columnName: z.string(),
  role: z.enum([
    "feature",
    "target",
    "identifier",
    "timestamp",
    "free_text",
    "ignore",
    "unknown"
  ]),
  semanticType: z.enum([
    "numeric",
    "categorical",
    "boolean",
    "date",
    "text",
    "id",
    "mixed",
    "empty"
  ]),
  confidence: z.enum(["low", "medium", "high"]),
  evidence: z.array(z.string()).max(5),
  risks: z.array(z.string()).max(5),
  recommendedActions: z.array(z.string()).max(5),
  status: z.literal("proposed")
});

const llmColumnAssumptionSchema = columnAssumptionSchema.omit({
  id: true,
  status: true
});

const preprocessingSuggestionActionSchema = z.string().trim().min(1).max(80);

const preprocessingSuggestionSchema = z.object({
  columnName: z.string(),
  action: preprocessingSuggestionActionSchema,
  reason: z.string(),
  implementation: z.string().optional(),
  alternatives: z.array(preprocessingSuggestionActionSchema).max(5)
});

const columnSelectionDecisionSchema = z.enum(["keep", "drop", "review"]);

const targetSuggestionSchema = z.object({
  columnName: z.string(),
  reason: z.string(),
  confidence: z.enum(["low", "medium", "high"])
});

const columnDecisionSchema = z.object({
  columnName: z.string(),
  decision: columnSelectionDecisionSchema,
  reason: z.string(),
  confidence: z.enum(["low", "medium", "high"]),
  alternatives: z.array(columnSelectionDecisionSchema).max(3)
});

const columnSelectionPlanSchema = z.object({
  datasetSummary: z.string(),
  targetSuggestion: targetSuggestionSchema.optional(),
  columnDecisions: z.array(columnDecisionSchema),
  globalWarnings: z.array(z.string()).max(8),
  nextQuestions: z.array(z.string()).max(5)
});

const preprocessingRequestSchema = z.object({
  profile: z.lazy(() => reviewProfileSchema),
  targetColumn: z.string(),
  keptColumns: z.array(z.string())
});

const preprocessingPlanSchema = z.object({
  datasetSummary: z.string(),
  assumptions: z.array(columnAssumptionSchema),
  preprocessingSuggestions: z.array(preprocessingSuggestionSchema),
  decisions: z.array(
    z.object({
      id: z.string(),
      type: z.enum([
        "assumption",
        "mapping",
        "normalization",
        "validation",
        "exclusion"
      ]),
      target: z.string(),
      decision: z.string(),
      reason: z.string(),
      confidence: z.enum(["low", "medium", "high"]),
      evidence: z.array(z.string()).max(5),
      alternatives: z.array(z.string()).max(5),
      relatedAssumptionIds: z.array(z.string()).max(5),
      relatedPreprocessingStepIds: z.array(z.string()).max(5)
    })
  ),
  globalWarnings: z.array(z.string()).max(8),
  nextQuestions: z.array(z.string()).max(5)
});
const llmReviewSchema = z.object({
  datasetSummary: z.string(),
  assumptions: z.array(llmColumnAssumptionSchema),
  globalWarnings: z.array(z.string()).max(8),
  nextQuestions: z.array(z.string()).max(5)
});
const llmPreprocessingReviewSchema = z.object({
  preprocessingSuggestions: z.array(preprocessingSuggestionSchema)
});

const reviewProfileSchema = z.object({
  fileName: z.string(),
  fileSizeBytes: z.number(),
  parsedRowCount: z.number(),
  columnCount: z.number(),
  warnings: z.array(z.string()),
  columns: z.array(
    z.object({
      name: z.string(),
      inferredType: z.string(),
      missingCount: z.number(),
      missingPercent: z.number(),
      nonMissingCount: z.number(),
      uniqueCount: z.number(),
      uniqueRatio: z.number(),
      topValues: z.array(
        z.object({
          value: z.string(),
          count: z.number(),
          percent: z.number()
        })
      ),
      sampleValues: z.array(z.string()),
      profilingNotes: z.array(
        z.object({
          code: z.string(),
          message: z.string(),
          affectedCount: z.number()
        })
      )
    })
  ),
  proposedPreprocessingSteps: z.array(
    z.object({
      id: z.string(),
      operation: z.enum([
        "trim_whitespace",
        "normalize_boolean",
        "standardize_missing_tokens"
      ]),
      target: z.string(),
      description: z.string()
    })
  ),
  previewRows: z.array(z.record(z.string(), z.unknown())).max(20)
});

type ReviewProfile = z.infer<typeof reviewProfileSchema>;
type PreprocessingPlan = z.infer<typeof preprocessingPlanSchema>;
type LlmReview = z.infer<typeof llmReviewSchema>;
type LlmPreprocessingReview = z.infer<typeof llmPreprocessingReviewSchema>;
type PreprocessingAction = z.infer<typeof preprocessingSuggestionActionSchema>;
type ColumnSelectionPlan = z.infer<typeof columnSelectionPlanSchema>;
type ColumnSelectionDecision = z.infer<typeof columnSelectionDecisionSchema>;

function extractJsonObject(text: string) {
  const fencedMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fencedMatch?.[1] ?? text;
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");

  if (start === -1 || end === -1 || end <= start) return null;
  return candidate.slice(start, end + 1);
}

function getJsonPayload(value: unknown): unknown {
  if (
    value &&
    typeof value === "object" &&
    "choices" in value &&
    Array.isArray((value as { choices?: unknown }).choices)
  ) {
    const choice = (value as { choices: unknown[] }).choices[0];
    if (choice && typeof choice === "object") {
      const message = (choice as { message?: { content?: unknown } }).message;
      if (typeof message?.content === "string") {
        const nested = extractJsonObject(message.content);
        if (nested) return JSON.parse(nested);
      }
    }
  }

  return value;
}

function roleForProfileColumn(
  column: ReviewProfile["columns"][number]
): PreprocessingPlan["assumptions"][number]["role"] {
  const name = column.name.toLowerCase();
  const notes = new Set(column.profilingNotes.map((note) => note.code));

  if (column.inferredType === "empty") return "ignore";
  if (notes.has("likely_identifier") || /(^|_)(id|uuid|guid)($|_)/.test(name))
    return "identifier";
  if (
    column.inferredType === "date" ||
    /(date|time|created|updated|timestamp)/.test(name)
  ) {
    return "timestamp";
  }
  if (/(target|label|outcome|churn|converted|conversion)/.test(name))
    return "target";
  if (column.inferredType === "string" && column.uniqueRatio > 0.75)
    return "free_text";
  return "feature";
}

function semanticTypeForProfileColumn(
  column: ReviewProfile["columns"][number]
): PreprocessingPlan["assumptions"][number]["semanticType"] {
  const role = roleForProfileColumn(column);

  if (role === "identifier") return "id";
  if (role === "free_text") return "text";
  if (column.inferredType === "number") return "numeric";
  if (column.inferredType === "boolean") return "boolean";
  if (column.inferredType === "date") return "date";
  if (column.inferredType === "mixed") return "mixed";
  if (column.inferredType === "empty") return "empty";
  return "categorical";
}

function confidenceForProfileColumn(
  column: ReviewProfile["columns"][number]
): PreprocessingPlan["assumptions"][number]["confidence"] {
  if (column.profilingNotes.length > 0 || column.inferredType === "mixed")
    return "medium";
  return "high";
}

function chooseFallbackTarget(profile: ReviewProfile) {
  const columns = profile.columns.filter(
    (column) => roleForProfileColumn(column) !== "identifier"
  );
  const nameMatched = columns.find((column) =>
    /(target|label|outcome|churn|converted|conversion|survived|class|status|result|score|price|amount|total|sales|revenue)$/i.test(
      column.name
    )
  );
  if (nameMatched) return nameMatched.name;

  const binaryColumn = columns.find(
    (column) =>
      column.uniqueCount === 2 &&
      column.missingPercent < 10 &&
      column.inferredType !== "empty"
  );
  if (binaryColumn) return binaryColumn.name;

  const lastUsableColumn = [...columns]
    .reverse()
    .find(
      (column) =>
        column.inferredType !== "empty" &&
        !column.profilingNotes.some((note) => note.code === "constant_column")
    );

  return lastUsableColumn?.name;
}

function plausibleColumnDecisions(
  column: ReviewProfile["columns"][number],
  targetColumn?: string
): ColumnSelectionDecision[] {
  const decisions: ColumnSelectionDecision[] = ["keep", "drop", "review"];
  if (column.name === targetColumn) return ["review", "keep"];
  return decisions;
}

function defaultColumnDecision(
  column: ReviewProfile["columns"][number],
  targetColumn?: string
): ColumnSelectionPlan["columnDecisions"][number] {
  const role = roleForProfileColumn(column);
  const isTarget = column.name === targetColumn;
  let decision: ColumnSelectionDecision = "keep";
  let reason = "Column may provide useful model signal.";
  let confidence: ColumnSelectionPlan["columnDecisions"][number]["confidence"] =
    "medium";

  if (isTarget) {
    decision = "review";
    reason =
      "Selected as the target candidate, so it should not be used as a feature.";
    confidence = "high";
  } else if (
    role === "identifier" ||
    role === "ignore" ||
    column.profilingNotes.some(
      (note) => note.code === "empty_column" || note.code === "constant_column"
    )
  ) {
    decision = "drop";
    reason =
      role === "identifier"
        ? "Likely identifier; including it may leak row identity."
        : "Column appears empty, constant, or otherwise low value.";
    confidence = "high";
  } else if (role === "free_text" || role === "unknown") {
    decision = "review";
    reason = "Column may need user judgment before keeping or dropping.";
  }

  return {
    columnName: column.name,
    decision,
    reason,
    confidence,
    alternatives: plausibleColumnDecisions(column, targetColumn).filter(
      (candidate) => candidate !== decision
    )
  };
}

function buildFallbackColumnSelectionPlan(profile: ReviewProfile) {
  const targetColumn = chooseFallbackTarget(profile);
  return {
    datasetSummary: `${profile.fileName} has ${profile.parsedRowCount.toLocaleString()} rows and ${profile.columnCount} columns.`,
    targetSuggestion: targetColumn
      ? {
          columnName: targetColumn,
          reason:
            "Selected from profile statistics as the most likely target candidate.",
          confidence: "medium" as const
        }
      : undefined,
    columnDecisions: profile.columns.map((column) =>
      defaultColumnDecision(column, targetColumn)
    ),
    globalWarnings: [
      "AI column selection response could not be parsed. Showing a deterministic fallback from profile statistics.",
      ...profile.warnings
    ].slice(0, 8),
    nextQuestions: ["Confirm the target column before preprocessing."]
  } satisfies ColumnSelectionPlan;
}

function normalizeColumnSelectionPlan(
  profile: ReviewProfile,
  plan: ColumnSelectionPlan
) {
  const columnsByName = new Map(
    profile.columns.map((column) => [column.name, column])
  );
  const targetColumn = columnsByName.has(
    plan.targetSuggestion?.columnName ?? ""
  )
    ? plan.targetSuggestion?.columnName
    : chooseFallbackTarget(profile);
  const seen = new Set<string>();
  const columnDecisions: ColumnSelectionPlan["columnDecisions"] = [];

  for (const decision of plan.columnDecisions) {
    const column = columnsByName.get(decision.columnName);
    if (!column || seen.has(decision.columnName)) continue;

    const plausible = plausibleColumnDecisions(column, targetColumn);
    const fallback = defaultColumnDecision(column, targetColumn);
    const selected = plausible.includes(decision.decision)
      ? decision.decision
      : fallback.decision;

    seen.add(decision.columnName);
    columnDecisions.push({
      ...decision,
      decision: selected,
      reason: decision.reason || fallback.reason,
      alternatives: Array.from(
        new Set([
          ...decision.alternatives.filter((item) => plausible.includes(item)),
          ...fallback.alternatives
        ])
      ).filter((item) => item !== selected)
    });
  }

  for (const column of profile.columns) {
    if (!seen.has(column.name)) {
      columnDecisions.push(defaultColumnDecision(column, targetColumn));
    }
  }

  return {
    ...plan,
    targetSuggestion: targetColumn
      ? {
          columnName: targetColumn,
          reason:
            plan.targetSuggestion?.columnName === targetColumn
              ? plan.targetSuggestion.reason
              : "Selected from profile statistics as the most likely target candidate.",
          confidence:
            plan.targetSuggestion?.columnName === targetColumn
              ? plan.targetSuggestion.confidence
              : ("medium" as const)
        }
      : undefined,
    columnDecisions
  } satisfies ColumnSelectionPlan;
}

function buildDecisionTrace(
  profile: ReviewProfile,
  assumptions: PreprocessingPlan["assumptions"]
) {
  const assumptionDecisions = assumptions.map((assumption) => ({
    id: `decision:${assumption.id}`,
    type: "assumption" as const,
    target: assumption.columnName,
    decision: `Treat ${assumption.columnName} as ${assumption.role}.`,
    reason:
      assumption.evidence[0] ??
      "The column role was selected from the compact dataset profile.",
    confidence: assumption.confidence,
    evidence: assumption.evidence.slice(0, 5),
    alternatives:
      assumption.role === "feature"
        ? ["target", "ignore", "unknown"]
        : ["feature", "unknown"],
    relatedAssumptionIds: [assumption.id],
    relatedPreprocessingStepIds: []
  }));

  const preprocessingDecisions = profile.proposedPreprocessingSteps.map(
    (step) => ({
      id: `decision:step:${step.id}`,
      type: "normalization" as const,
      target: step.target,
      decision: step.description,
      reason:
        "The CSV profiler detected a deterministic cleanup opportunity in the sampled values.",
      confidence: "high" as const,
      evidence: [step.description],
      alternatives: ["Leave raw values unchanged until manually confirmed."],
      relatedAssumptionIds: [],
      relatedPreprocessingStepIds: [step.id]
    })
  );

  return [...assumptionDecisions, ...preprocessingDecisions];
}

function buildPreprocessingPlan(
  profile: ReviewProfile,
  review: LlmReview & {
    assumptions: PreprocessingPlan["assumptions"];
    preprocessingSuggestions: PreprocessingPlan["preprocessingSuggestions"];
  }
): PreprocessingPlan {
  return {
    ...review,
    decisions: buildDecisionTrace(profile, review.assumptions)
  };
}

function buildLocalAssumption(
  column: ReviewProfile["columns"][number]
): PreprocessingPlan["assumptions"][number] {
  const role = roleForProfileColumn(column);

  return {
    id: `column:${column.name}`,
    columnName: column.name,
    role,
    semanticType: semanticTypeForProfileColumn(column),
    confidence: confidenceForProfileColumn(column),
    evidence: [
      `Inferred type: ${column.inferredType}.`,
      `Missing values: ${column.missingPercent.toFixed(1)}%.`,
      `Unique ratio: ${column.uniqueRatio.toFixed(2)}.`
    ],
    risks: column.profilingNotes.slice(0, 3).map((note) => note.message),
    recommendedActions:
      role === "identifier"
        ? ["Exclude from model features unless the identifier carries meaning."]
        : role === "ignore"
          ? ["Keep excluded unless the column is populated in the source data."]
          : ["Confirm this role against the dataset objective."],
    status: "proposed"
  };
}

function uniqueActions(actions: PreprocessingAction[]) {
  return Array.from(new Set(actions));
}

const PREPROCESSING_ACTIONS = {
  none: "No step",
  drop: "Drop column",
  fillMean: "Fill missing values with the mean",
  fillMedian: "Fill missing values with the median",
  fillMode: "Fill missing values with the most common value",
  oneHotEncode: "One-hot encode categories",
  trimWhitespace: "Trim whitespace",
  standardizeMissing: "Standardize missing-value tokens",
  normalizeBoolean: "Normalize boolean values",
  splitName: "Split full names into parts"
} as const;

function isLikelyNameColumn(column: ReviewProfile["columns"][number]) {
  const name = column.name.toLowerCase();
  return (
    /(name|full_name|fullname|passengername|customer_name)/.test(name) ||
    column.sampleValues.some((value) => {
      const trimmed = value.trim();
      return /^[A-Z][a-z]+(?:\s+[A-Z][a-z.'-]+)+$/.test(trimmed);
    })
  );
}

function plausiblePreprocessingActions(
  column: ReviewProfile["columns"][number],
  assumption: PreprocessingPlan["assumptions"][number]
) {
  const actions: PreprocessingAction[] = [PREPROCESSING_ACTIONS.none];

  if (assumption.role === "identifier" || assumption.role === "ignore") {
    actions.push(PREPROCESSING_ACTIONS.drop);
  }
  if (
    column.profilingNotes.some(
      (note) => note.code === "empty_column" || note.code === "constant_column"
    )
  ) {
    actions.push(PREPROCESSING_ACTIONS.drop);
  }
  if (column.profilingNotes.some((note) => note.code === "missing_like_tokens"))
    actions.push(PREPROCESSING_ACTIONS.standardizeMissing);
  if (
    column.profilingNotes.some(
      (note) => note.code === "leading_trailing_whitespace"
    )
  ) {
    actions.push(PREPROCESSING_ACTIONS.trimWhitespace);
  }
  if (column.inferredType === "boolean")
    actions.push(PREPROCESSING_ACTIONS.normalizeBoolean);
  if (column.missingCount > 0 && column.inferredType === "number") {
    actions.push(
      PREPROCESSING_ACTIONS.fillMean,
      PREPROCESSING_ACTIONS.fillMedian
    );
  }
  if (
    column.missingCount > 0 &&
    (column.inferredType === "string" || column.inferredType === "boolean")
  ) {
    actions.push(PREPROCESSING_ACTIONS.fillMode);
  }
  if (
    column.inferredType === "string" &&
    column.uniqueCount > 1 &&
    column.uniqueRatio <= 0.2
  ) {
    actions.push(PREPROCESSING_ACTIONS.oneHotEncode);
  }
  if (isLikelyNameColumn(column)) {
    actions.push(PREPROCESSING_ACTIONS.splitName, PREPROCESSING_ACTIONS.drop);
  }

  return uniqueActions(actions);
}

function defaultPreprocessingSuggestion(
  column: ReviewProfile["columns"][number],
  assumption: PreprocessingPlan["assumptions"][number]
): PreprocessingPlan["preprocessingSuggestions"][number] {
  const actions = plausiblePreprocessingActions(column, assumption);
  let action = actions[0] ?? PREPROCESSING_ACTIONS.none;
  let reason = "No specific preprocessing needed from the current profile.";

  if (assumption.role === "identifier" || assumption.role === "ignore") {
    action = PREPROCESSING_ACTIONS.drop;
    reason = "The column is not expected to be useful as a model feature.";
  } else if (isLikelyNameColumn(column)) {
    action = PREPROCESSING_ACTIONS.splitName;
    reason =
      "Sample values look like full names, which can be split into reusable parts.";
  } else if (
    column.profilingNotes.some((note) => note.code === "empty_column")
  ) {
    action = PREPROCESSING_ACTIONS.drop;
    reason = "The column has no non-empty sampled values.";
  } else if (
    column.profilingNotes.some((note) => note.code === "constant_column")
  ) {
    action = PREPROCESSING_ACTIONS.drop;
    reason = "The column is constant in the sampled rows.";
  } else if (
    column.profilingNotes.some((note) => note.code === "missing_like_tokens")
  ) {
    action = PREPROCESSING_ACTIONS.standardizeMissing;
    reason = "The column contains tokens that commonly represent missingness.";
  } else if (
    column.profilingNotes.some(
      (note) => note.code === "leading_trailing_whitespace"
    )
  ) {
    action = PREPROCESSING_ACTIONS.trimWhitespace;
    reason = "Sample values include leading or trailing whitespace.";
  } else if (column.inferredType === "boolean") {
    action = PREPROCESSING_ACTIONS.normalizeBoolean;
    reason = "Boolean-like values should use one consistent representation.";
  } else if (column.missingCount > 0 && column.inferredType === "number") {
    action =
      column.missingPercent >= 20
        ? PREPROCESSING_ACTIONS.fillMedian
        : PREPROCESSING_ACTIONS.fillMean;
    reason =
      column.missingPercent >= 20
        ? "Median imputation is robust for numeric columns with material missingness."
        : "Mean imputation is a simple baseline for low numeric missingness.";
  } else if (column.missingCount > 0 && column.inferredType === "string") {
    action = PREPROCESSING_ACTIONS.fillMode;
    reason = "Categorical missing values can use the most frequent value.";
  } else if (
    column.inferredType === "string" &&
    column.uniqueCount > 1 &&
    column.uniqueRatio <= 0.2
  ) {
    action = PREPROCESSING_ACTIONS.oneHotEncode;
    reason = "Low-cardinality categorical values are suitable for encoding.";
  }

  return {
    columnName: column.name,
    action,
    reason,
    implementation:
      action === PREPROCESSING_ACTIONS.none
        ? "Leave the column unchanged."
        : `${action} before modeling.`,
    alternatives: actions.filter((candidate) => candidate !== action)
  };
}

function normalizeLlmReview(
  profile: ReviewProfile,
  review: LlmReview,
  preprocessingReview: LlmPreprocessingReview
): LlmReview & {
  assumptions: PreprocessingPlan["assumptions"];
  preprocessingSuggestions: PreprocessingPlan["preprocessingSuggestions"];
} {
  const columnsByName = new Map(
    profile.columns.map((column) => [column.name, column])
  );
  const seen = new Set<string>();
  const assumptions: PreprocessingPlan["assumptions"] = [];

  for (const assumption of review.assumptions) {
    if (seen.has(assumption.columnName)) continue;
    if (!columnsByName.has(assumption.columnName)) continue;

    seen.add(assumption.columnName);
    assumptions.push({
      ...assumption,
      id: `column:${assumption.columnName}`,
      status: "proposed"
    });
  }

  for (const column of profile.columns) {
    if (!seen.has(column.name)) assumptions.push(buildLocalAssumption(column));
  }

  const assumptionsByColumn = new Map(
    assumptions.map((assumption) => [assumption.columnName, assumption])
  );
  const suggestionSeen = new Set<string>();
  const preprocessingSuggestions: PreprocessingPlan["preprocessingSuggestions"] =
    [];

  for (const suggestion of preprocessingReview.preprocessingSuggestions) {
    if (suggestionSeen.has(suggestion.columnName)) continue;
    const column = columnsByName.get(suggestion.columnName);
    if (!column) continue;

    const assumption =
      assumptionsByColumn.get(column.name) ?? buildLocalAssumption(column);
    const fallback = defaultPreprocessingSuggestion(column, assumption);
    const action = suggestion.action || fallback.action;
    const alternatives = uniqueActions([
      action,
      ...suggestion.alternatives,
      ...fallback.alternatives
    ]).filter((candidate) => candidate !== action);

    suggestionSeen.add(suggestion.columnName);
    preprocessingSuggestions.push({
      columnName: suggestion.columnName,
      action,
      reason: suggestion.reason || fallback.reason,
      implementation: suggestion.implementation || fallback.implementation,
      alternatives
    });
  }

  for (const column of profile.columns) {
    if (suggestionSeen.has(column.name)) continue;
    const assumption =
      assumptionsByColumn.get(column.name) ?? buildLocalAssumption(column);
    preprocessingSuggestions.push(
      defaultPreprocessingSuggestion(column, assumption)
    );
  }

  return {
    ...review,
    assumptions,
    preprocessingSuggestions
  };
}

function buildFallbackPreprocessingPlan(profile: ReviewProfile) {
  const fallbackTarget = chooseFallbackTarget(profile);
  const assumptions = profile.columns.map((column) => {
    const assumption = buildLocalAssumption(column);
    if (column.name !== fallbackTarget) return assumption;

    return {
      ...assumption,
      role: "target" as const,
      evidence: [
        `Selected as a fallback target candidate from ${column.uniqueCount.toLocaleString()} unique value${column.uniqueCount === 1 ? "" : "s"}.`,
        ...assumption.evidence
      ].slice(0, 5),
      recommendedActions: [
        "Confirm this is the outcome column before accepting preprocessing."
      ]
    };
  });

  return buildPreprocessingPlan(profile, {
    datasetSummary: `${profile.fileName} has ${profile.parsedRowCount.toLocaleString()} rows and ${profile.columnCount} columns.`,
    assumptions,
    preprocessingSuggestions: profile.columns.map((column) => {
      const assumption =
        assumptions.find((item) => item.columnName === column.name) ??
        buildLocalAssumption(column);
      return defaultPreprocessingSuggestion(column, assumption);
    }),
    globalWarnings: [
      "AI review response could not be parsed. Showing a local fallback review from deterministic profile statistics.",
      ...profile.warnings
    ].slice(0, 8),
    nextQuestions: [
      "Which column is the target outcome?",
      "Are identifier columns safe to exclude from model features?"
    ]
  });
}

function parseLlmReviewText(text: string) {
  const jsonText = extractJsonObject(text);
  if (!jsonText) {
    return {
      success: false as const,
      error: "The model did not return a JSON object."
    };
  }

  try {
    const parsed = getJsonPayload(JSON.parse(jsonText));
    const validated = llmReviewSchema.safeParse(parsed);

    if (!validated.success) {
      return {
        success: false as const,
        error: z.prettifyError(validated.error)
      };
    }

    return {
      success: true as const,
      review: validated.data
    };
  } catch (error) {
    return {
      success: false as const,
      error:
        error instanceof Error ? error.message : "The JSON could not be parsed."
    };
  }
}

function parseColumnSelectionText(text: string) {
  const jsonText = extractJsonObject(text);
  if (!jsonText) {
    return {
      success: false as const,
      error: "The model did not return a JSON object."
    };
  }

  try {
    const parsed = getJsonPayload(JSON.parse(jsonText));
    const validated = columnSelectionPlanSchema.safeParse(parsed);

    if (!validated.success) {
      return {
        success: false as const,
        error: z.prettifyError(validated.error)
      };
    }

    return {
      success: true as const,
      plan: validated.data
    };
  } catch (error) {
    return {
      success: false as const,
      error:
        error instanceof Error ? error.message : "The JSON could not be parsed."
    };
  }
}

function parseLlmPreprocessingText(text: string) {
  const jsonText = extractJsonObject(text);
  if (!jsonText) {
    return {
      success: false as const,
      error: "The model did not return a JSON object."
    };
  }

  try {
    const parsed = getJsonPayload(JSON.parse(jsonText));
    const validated = llmPreprocessingReviewSchema.safeParse(parsed);

    if (!validated.success) {
      return {
        success: false as const,
        error: z.prettifyError(validated.error)
      };
    }

    return {
      success: true as const,
      review: validated.data
    };
  } catch (error) {
    return {
      success: false as const,
      error:
        error instanceof Error ? error.message : "The JSON could not be parsed."
    };
  }
}

function buildReviewPrompt(profile: ReviewProfile) {
  return `Return JSON only. Do not use Markdown. Do not include a choices array. Do not include prose before or after the JSON.

Use this exact JSON shape:
{
  "datasetSummary": "short plain English summary",
  "assumptions": [
    {
      "columnName": "exact column name from input",
      "role": "feature | target | identifier | timestamp | free_text | ignore | unknown",
      "semanticType": "numeric | categorical | boolean | date | text | id | mixed | empty",
      "confidence": "low | medium | high",
      "evidence": ["short evidence from profile"],
      "risks": ["short risk, or empty array"],
      "recommendedActions": ["short recommended action"]
    }
  ],
  "globalWarnings": ["dataset-level warning, or empty array"],
  "nextQuestions": ["question for the user, or empty array"]
}

Example response:
{
  "datasetSummary": "customers.csv has 500 rows and 6 columns. The profile suggests an identifier, timestamp, and candidate target column.",
  "assumptions": [
    {
      "columnName": "customer_id",
      "role": "identifier",
      "semanticType": "id",
      "confidence": "high",
      "evidence": ["The column name contains id.", "The unique ratio is 1.00."],
      "risks": ["Using this as a feature may leak row identity."],
      "recommendedActions": ["Exclude from model features unless it encodes useful grouping."]
    },
    {
      "columnName": "full_name",
      "role": "free_text",
      "semanticType": "text",
      "confidence": "medium",
      "evidence": ["Sample values look like person names."],
      "risks": ["Raw names can create high-cardinality sparse features."],
      "recommendedActions": ["Split into useful parts or drop if identity leakage is a concern."]
    }
  ],
  "globalWarnings": [],
  "nextQuestions": ["Which column is the target outcome?"]
}

Rules:
- Use every columnName exactly as provided in the input.
- Prefer "unknown" with low confidence when evidence is weak.
- Identifier columns should usually not be model features.
- Target columns should not be treated as features.
- Base conclusions only on profile statistics, profiling notes, top values, sample values, proposed preprocessing steps, and preview rows.

Input profile:
${JSON.stringify(profile)}`;
}

function buildColumnSelectionPrompt(profile: ReviewProfile) {
  return `Return JSON only. Do not use Markdown. Do not include prose before or after the JSON.

Choose the target column and decide which columns to keep before preprocessing. Do not suggest preprocessing steps.

Use this exact JSON shape:
{
  "datasetSummary": "short plain English summary",
  "targetSuggestion": {
    "columnName": "exact column name from input",
    "reason": "why this is likely the target",
    "confidence": "low | medium | high"
  },
  "columnDecisions": [
    {
      "columnName": "exact column name from input",
      "decision": "keep | drop | review",
      "reason": "why to keep, drop, or review",
      "confidence": "low | medium | high",
      "alternatives": ["other plausible decisions"]
    }
  ],
  "globalWarnings": ["dataset-level warning, or empty array"],
  "nextQuestions": ["question for the user, or empty array"]
}

Rules:
- Include one columnDecisions entry for every input column.
- Use every columnName exactly as provided in the input.
- The target column decision should be "review", not "keep".
- Drop row identifiers, empty columns, constant columns, and clearly irrelevant leakage columns.
- Use review for text columns that may be useful only after feature extraction, such as names or tickets.
- Do not suggest fill, encode, split, normalize, or any preprocessing action in this call.

Input profile:
${JSON.stringify(profile)}`;
}

function buildPreprocessingPrompt(profile: ReviewProfile, review: LlmReview) {
  return `Return compact JSON only. Do not use Markdown. Do not include prose before or after the JSON.

Use this exact JSON shape:
{"preprocessingSuggestions":[{"columnName":"exact column name from input","action":"short step name","reason":"under 18 words","implementation":"under 18 words","alternatives":["step names only"]}]}

Rules:
- Include one preprocessingSuggestions entry for every input column.
- Keep every reason and implementation under 18 words.
- Keep alternatives to at most 3 short strings.
- Use only these actions: No step, Trim whitespace, Lowercase text, Standardize missing-value tokens, Normalize boolean values, Fill missing values with the mean, Fill missing values with the median, Fill missing values with the most common value, One-hot encode categories, Split full names into parts, Drop column.
- Prefer specific, dataset-aware recommendations over generic labels.
- Keep recommendations practical and implementable from the available column values.
- Do not suggest numeric imputation for boolean or string columns.
- Do not suggest categorical mode imputation for numeric columns.
- Use drop for identifiers, empty columns, constant columns, or clearly irrelevant high-cardinality text.
- Keep target columns as "No step".

Column assumptions:
${JSON.stringify(review.assumptions)}

Input profile:
${JSON.stringify(profile)}`;
}

function buildColumnPreprocessingPrompt({
  profile,
  targetColumn,
  keptColumns
}: z.infer<typeof preprocessingRequestSchema>) {
  const kept = new Set(keptColumns);
  const scopedProfile = {
    ...profile,
    columns: profile.columns.filter(
      (column) => column.name === targetColumn || kept.has(column.name)
    ),
    previewRows: profile.previewRows.map((row) =>
      Object.fromEntries(
        Object.entries(row).filter(
          ([columnName]) => columnName === targetColumn || kept.has(columnName)
        )
      )
    )
  };

  return `Return compact JSON only. Do not use Markdown. Do not include prose before or after the JSON.

Suggest preprocessing only for kept feature columns. The target column is ${JSON.stringify(targetColumn)} and must not receive a preprocessing step.

Use this exact JSON shape:
{"preprocessingSuggestions":[{"columnName":"exact kept column name","action":"short step name","reason":"under 18 words","implementation":"under 18 words","alternatives":["step names only"]}]}

Rules:
- Include one preprocessingSuggestions entry for every kept feature column.
- Do not include dropped columns.
- Do not include the target column.
- Keep every reason and implementation under 18 words.
- Keep alternatives to at most 3 short strings.
- Use only these actions: No step, Trim whitespace, Lowercase text, Standardize missing-value tokens, Normalize boolean values, Fill missing values with the mean, Fill missing values with the median, Fill missing values with the most common value, One-hot encode categories, Split full names into parts, Drop column.
- Prefer specific, dataset-aware recommendations over generic labels.
- Keep recommendations practical and implementable from the available column values.
- Do not suggest numeric imputation for boolean or string columns.
- Do not suggest categorical mode imputation for numeric columns.
- Use "No step" only when the column should pass through unchanged.

Kept columns:
${JSON.stringify(keptColumns)}

Scoped input profile:
${JSON.stringify(scopedProfile)}`;
}

function buildFallbackColumnPreprocessingResponse(
  profile: ReviewProfile,
  keptColumns: string[],
  warning?: string
) {
  return {
    preprocessingSuggestions: keptColumns
      .map((columnName) => {
        const column = profile.columns.find((item) => item.name === columnName);
        if (!column) return null;
        return defaultPreprocessingSuggestion(
          column,
          buildLocalAssumption(column)
        );
      })
      .filter(Boolean),
    globalWarnings: warning ? [warning] : [],
    nextQuestions: []
  };
}

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

async function handleAiReview(request: Request, env: Env) {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const body = await request.json().catch(() => null);
  const parsedProfile = reviewProfileSchema.safeParse(
    (body as { profile?: unknown } | null)?.profile
  );

  if (!parsedProfile.success) {
    return Response.json(
      { error: "Request must include a valid compact dataset profile." },
      { status: 400 }
    );
  }

  const workersai = createWorkersAI({ binding: env.AI });
  const model = env.AI_MODEL || DEFAULT_AI_MODEL;
  const profile = parsedProfile.data;

  const result = await generateText({
    model: workersai(model),
    ...JSON_GENERATION_SETTINGS,
    system:
      "You review compact CSV profiles for preprocessing planning. You output JSON only.",
    prompt: buildReviewPrompt(profile)
  });
  const parsedReview = parseLlmReviewText(result.text);

  if (!parsedReview.success) {
    console.warn("AI review response could not be parsed", {
      error: parsedReview.error,
      generatedText: result.text.slice(0, 500)
    });

    return Response.json(buildFallbackPreprocessingPlan(profile));
  }

  const preprocessingResult = await generateText({
    model: workersai(model),
    ...JSON_GENERATION_SETTINGS,
    system:
      "You suggest CSV preprocessing steps. You output one JSON object only.",
    prompt: buildPreprocessingPrompt(profile, parsedReview.review)
  });
  const parsedPreprocessing = parseLlmPreprocessingText(
    preprocessingResult.text
  );

  if (!parsedPreprocessing.success) {
    console.warn("AI preprocessing response could not be parsed", {
      error: parsedPreprocessing.error,
      generatedText: preprocessingResult.text.slice(0, 500)
    });
  }

  return Response.json(
    buildPreprocessingPlan(
      profile,
      normalizeLlmReview(
        profile,
        parsedReview.review,
        parsedPreprocessing.success
          ? parsedPreprocessing.review
          : { preprocessingSuggestions: [] }
      )
    )
  );
}

async function handleColumnSelection(request: Request, env: Env) {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const body = await request.json().catch(() => null);
  const parsedProfile = reviewProfileSchema.safeParse(
    (body as { profile?: unknown } | null)?.profile
  );

  if (!parsedProfile.success) {
    return Response.json(
      { error: "Request must include a valid compact dataset profile." },
      { status: 400 }
    );
  }

  const workersai = createWorkersAI({ binding: env.AI });
  const model = env.AI_MODEL || DEFAULT_AI_MODEL;
  const profile = parsedProfile.data;
  const result = await generateText({
    model: workersai(model),
    ...JSON_GENERATION_SETTINGS,
    system:
      "You decide which CSV columns should be kept before preprocessing. You output JSON only.",
    prompt: buildColumnSelectionPrompt(profile)
  });
  const parsedPlan = parseColumnSelectionText(result.text);

  if (!parsedPlan.success) {
    console.warn("AI column selection response could not be parsed", {
      error: parsedPlan.error,
      generatedText: result.text.slice(0, 500)
    });

    return Response.json(buildFallbackColumnSelectionPlan(profile));
  }

  return Response.json(normalizeColumnSelectionPlan(profile, parsedPlan.plan));
}

async function handlePreprocessingReview(request: Request, env: Env) {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const body = await request.json().catch(() => null);
  const parsedRequest = preprocessingRequestSchema.safeParse(body);

  if (!parsedRequest.success) {
    return Response.json(
      {
        error:
          "Request must include a valid profile, targetColumn, and keptColumns."
      },
      { status: 400 }
    );
  }

  const { profile, targetColumn, keptColumns } = parsedRequest.data;
  const allowedColumns = new Set(profile.columns.map((column) => column.name));
  const sanitizedKeptColumns = keptColumns.filter(
    (columnName) =>
      allowedColumns.has(columnName) && columnName !== targetColumn
  );
  const workersai = createWorkersAI({ binding: env.AI });
  const model = env.AI_MODEL || DEFAULT_AI_MODEL;
  const result = await generateText({
    model: workersai(model),
    ...JSON_GENERATION_SETTINGS,
    system:
      "You suggest preprocessing steps for selected CSV columns. You output JSON only.",
    prompt: buildColumnPreprocessingPrompt({
      profile,
      targetColumn,
      keptColumns: sanitizedKeptColumns
    })
  });
  const parsedPreprocessing = parseLlmPreprocessingText(result.text);

  if (!parsedPreprocessing.success) {
    console.warn("AI preprocessing response could not be parsed", {
      error: parsedPreprocessing.error,
      generatedText: result.text.slice(0, 500)
    });

    return Response.json(
      buildFallbackColumnPreprocessingResponse(
        profile,
        sanitizedKeptColumns,
        "AI preprocessing response could not be parsed. Showing deterministic fallback suggestions."
      )
    );
  }

  const suggestionsByColumn = new Map(
    parsedPreprocessing.review.preprocessingSuggestions.map((suggestion) => [
      suggestion.columnName,
      suggestion
    ])
  );
  const preprocessingSuggestions = sanitizedKeptColumns
    .map((columnName) => {
      const column = profile.columns.find((item) => item.name === columnName);
      if (!column) return null;

      const fallback = defaultPreprocessingSuggestion(
        column,
        buildLocalAssumption(column)
      );
      const suggestion = suggestionsByColumn.get(columnName);
      if (!suggestion) return fallback;

      const action = suggestion.action || fallback.action;

      return {
        columnName,
        action,
        reason: suggestion.reason || fallback.reason,
        implementation: suggestion.implementation || fallback.implementation,
        alternatives: uniqueActions([
          ...suggestion.alternatives,
          ...fallback.alternatives
        ]).filter((item) => item !== action)
      };
    })
    .filter(Boolean);

  return Response.json({
    preprocessingSuggestions,
    globalWarnings: [],
    nextQuestions: []
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

If the user asks to schedule a task or reminder, use the schedule tool.
If the user asks about the active CSV, current preprocessing plan, next feature to inspect, leakage risks, or what to do next, call getActiveDatasetContext before answering.`,
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
        getActiveDatasetContext: tool({
          description:
            "Get the active CSV profile, target, kept features, dropped columns, and approved preprocessing choices from the browser.",
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
    const url = new URL(request.url);
    if (url.pathname === "/api/column-selection") {
      return await handleColumnSelection(request, env);
    }
    if (url.pathname === "/api/preprocessing-review") {
      return await handlePreprocessingReview(request, env);
    }
    if (url.pathname === "/api/ai-review") {
      return await handleAiReview(request, env);
    }

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
