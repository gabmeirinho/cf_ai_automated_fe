import { describe, expect, test } from "vitest";
import {
  buildAiReviewProfile,
  buildDatasetIntent,
  isTrainingRow,
  parseCsvFile,
  transformCsvFile,
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

describe("parseCsvFile", () => {
  test("profiles all rows instead of only the first 1,000", async () => {
    const rows = Array.from({ length: 1005 }, (_, index) => {
      const rowNumber = index + 1;
      return [
        `id-${rowNumber}`,
        rowNumber <= 1000 ? String(rowNumber) : "not-a-number",
        rowNumber <= 1000 ? "present" : ""
      ].join(",");
    });
    const file = new File(
      [`id,value,late_missing\n${rows.join("\n")}`],
      "full.csv",
      { type: "text/csv" }
    );

    const parsed = await parseCsvFile(file);
    const value = parsed.columns.find((column) => column.name === "value");
    const lateMissing = parsed.columns.find(
      (column) => column.name === "late_missing"
    );

    expect(parsed.parsedRowCount).toBe(1005);
    expect(parsed.previewRows).toHaveLength(20);
    expect(value?.inferredType).toBe("mixed");
    expect(lateMissing?.missingCount).toBe(5);
    expect(parsed.warnings).toContain("Profiled all 1,005 parsed rows.");
  });
});

describe("transformCsvFile", () => {
  test("keeps split export row counts aligned with the training predicate", async () => {
    const file = new File(
      [
        [
          "id,feature,target",
          "1,a,yes",
          "2,b,no",
          "3,c,yes",
          "4,d,no",
          "5,e,yes",
          "6,f,no",
          "7,g,yes"
        ].join("\n")
      ],
      "split.csv",
      { type: "text/csv" }
    );

    const result = await transformCsvFile(
      file,
      {
        targetColumn: "target",
        featureColumns: ["id", "feature"],
        preprocessingSteps: []
      },
      { trainRatio: 0.7, seed: 42 }
    );

    if (!("trainCsv" in result)) {
      throw new Error("Expected split export result.");
    }

    const expectedTrainCount = Array.from(
      { length: 7 },
      (_, index) => index
    ).filter((index) => isTrainingRow(index, 42, 0.7)).length;

    expect(result.trainRowCount).toBe(expectedTrainCount);
    expect(result.testRowCount).toBe(7 - expectedTrainCount);
    expect(result.trainRowCount + result.testRowCount).toBe(7);
  });

  test("applies multiple steps to one feature and keeps target unchanged", async () => {
    const file = new File(
      [
        [
          "Name,Embarked,Survived",
          '"Braund, Mr. Owen Harris",S,0',
          '"Cumings, Mrs. John Bradley",C,1'
        ].join("\n")
      ],
      "titanic.csv",
      { type: "text/csv" }
    );

    const result = await transformCsvFile(file, {
      targetColumn: "Survived",
      featureColumns: ["Name", "Embarked"],
      preprocessingSteps: [
        { columnName: "Name", action: "Trim whitespace" },
        { columnName: "Name", action: "Lowercase text" },
        { columnName: "Name", action: "Split full names into parts" },
        { columnName: "Embarked", action: "One-hot encode categories" }
      ]
    });
    const transformed =
      result as import("./csv-profile").CsvTransformationResult;

    expect(transformed.outputColumns).toEqual([
      "Name_first",
      "Name_last",
      "Embarked__c",
      "Embarked__s",
      "Survived"
    ]);
    expect(transformed.csv).toContain("owen,harris,0,1,0");
    expect(transformed.csv).toContain("john,bradley,1,0,1");
    expect(transformed.audit).toContain(
      "Target Survived was kept unchanged and excluded from feature transforms."
    );
  });

  test("exports only approved engineered features", async () => {
    const file = new File(
      ["age,income,churn\n20,1000,no\n40,4000,yes"],
      "features.csv",
      { type: "text/csv" }
    );

    const result = await transformCsvFile(file, {
      targetColumn: "churn",
      featureColumns: ["age", "income"],
      preprocessingSteps: [],
      engineeredFeatures: [
        {
          expression: {
            op: "ratio",
            numerator: "income",
            denominator: "age"
          },
          name: "fe_income_to_age",
          reason: "Normalizes income by age.",
          expectedBenefit: "Adds a scale-adjusted income signal.",
          warnings: []
        }
      ]
    });
    const transformed =
      result as import("./csv-profile").CsvTransformationResult;

    expect(transformed.outputColumns).toEqual([
      "age",
      "income",
      "fe_income_to_age",
      "churn"
    ]);
    expect(transformed.csv).toContain("20,1000,50,no");
    expect(transformed.audit).toContain("1 accepted engineered feature added.");
  });

  test("blocks target leakage in preprocessing steps", async () => {
    const file = new File(["age,churn\n32,yes"], "leak.csv", {
      type: "text/csv"
    });

    await expect(
      transformCsvFile(file, {
        targetColumn: "churn",
        featureColumns: ["age"],
        preprocessingSteps: [{ columnName: "churn", action: "Lowercase text" }]
      })
    ).rejects.toThrow("Target leakage blocked");
  });
});
