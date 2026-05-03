import { describe, expect, test } from "vitest";
import {
  buildAiReviewProfile,
  buildDatasetIntent,
  type ColumnAssumption,
  type DatasetSummary
} from "./csv-profile";

function summary(columnNames: string[]): DatasetSummary {
  return {
    fileName: "sample.csv",
    fileSizeBytes: 120,
    parsedRowCount: 20,
    columns: columnNames.map((name) => ({
      name,
      inferredType: name === "empty_col" ? "empty" : "string",
      missingCount: name === "empty_col" ? 20 : 0,
      nonMissingCount: name === "empty_col" ? 0 : 20,
      uniqueCount: name.endsWith("_id") ? 20 : 2,
      uniqueRatio: name.endsWith("_id") ? 1 : 0.1,
      topValues: [],
      sampleValues: [],
      profilingNotes: []
    })),
    previewRows: [],
    warnings: [],
    proposedPreprocessingSteps: [
      {
        id: "trim_whitespace",
        operation: "trim_whitespace",
        columns: columnNames,
        status: "proposed"
      }
    ]
  };
}

function assumption(
  id: string,
  columnName: string,
  role: ColumnAssumption["role"],
  status: ColumnAssumption["status"] = "accepted"
): ColumnAssumption {
  return {
    id,
    columnName,
    role,
    semanticType: role === "identifier" ? "id" : "categorical",
    confidence: "high",
    evidence: [],
    risks: [],
    recommendedActions: [],
    status
  };
}

describe("buildAiReviewProfile", () => {
  test("includes proposed preprocessing steps for decision inspection", () => {
    const profile = buildAiReviewProfile(summary(["name", "status"]));

    expect(profile.proposedPreprocessingSteps).toEqual([
      {
        id: "trim_whitespace",
        operation: "trim_whitespace",
        target: "name, status",
        description:
          "Trim whitespace: Remove leading and trailing whitespace from 2 columns."
      }
    ]);
  });
});

describe("buildDatasetIntent", () => {
  test("derives canonical intent from accepted assumptions only", () => {
    const intent = buildDatasetIntent(
      summary(["customer_id", "churned", "age"]),
      [
        assumption("id", "customer_id", "identifier"),
        assumption("target", "churned", "target"),
        assumption("feature", "age", "feature"),
        assumption("ignored", "debug_value", "feature", "rejected")
      ]
    );

    expect(intent.targetColumn).toBe("churned");
    expect(intent.identifierColumns).toEqual(["customer_id"]);
    expect(intent.featureColumns).toEqual(["age"]);
    expect(intent.rejectedAssumptionIds).toEqual(["ignored"]);
    expect(intent.conflicts).toEqual([]);
  });

  test("removes targets and identifiers from feature columns", () => {
    const intent = buildDatasetIntent(summary(["customer_id", "churned"]), [
      assumption("id", "customer_id", "identifier"),
      assumption("id-feature", "customer_id", "feature"),
      assumption("target", "churned", "target"),
      assumption("target-feature", "churned", "feature")
    ]);

    expect(intent.featureColumns).toEqual([]);
    expect(intent.conflicts).toEqual([
      "customer_id has multiple accepted roles: identifier, feature. Keep only the role that matches the dataset objective.",
      "churned has multiple accepted roles: target, feature. Keep only the role that matches the dataset objective."
    ]);
  });

  test("reports multiple accepted targets", () => {
    const intent = buildDatasetIntent(summary(["churned", "converted"]), [
      assumption("target-a", "churned", "target"),
      assumption("target-b", "converted", "target")
    ]);

    expect(intent.targetColumn).toBe("churned");
    expect(intent.conflicts).toContain(
      "Multiple target columns are accepted: churned, converted. Choose one target before generating modeling steps."
    );
  });

  test("blocks empty columns from being accepted as modeling fields", () => {
    const intent = buildDatasetIntent(summary(["empty_col"]), [
      assumption("empty-target", "empty_col", "target")
    ]);

    expect(intent.conflicts).toContain(
      "empty_col is empty but accepted as target."
    );
  });
});
