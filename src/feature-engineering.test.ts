import { describe, expect, test } from "vitest";
import {
  applyValidatedFeatureSuggestions,
  validateFeatureSuggestions,
  type FeatureValidationContext
} from "./feature-engineering";

function baseContext(
  overrides: Partial<FeatureValidationContext> = {}
): FeatureValidationContext {
  return {
    availableColumns: ["age", "income", "signup_date"],
    numericColumns: ["age", "income"],
    dateColumns: ["signup_date"],
    targetColumn: "churn",
    blockedColumns: [],
    existingColumns: ["age", "income", "signup_date", "churn"],
    ...overrides
  };
}

describe("validateFeatureSuggestions", () => {
  test("accepts valid suggestions and sanitizes names", () => {
    const result = validateFeatureSuggestions(
      [
        {
          expression: { op: "ratio", numerator: "income", denominator: "age" },
          name: "income / age",
          reason: "Normalizes income by age.",
          expectedBenefit: "Can reduce scale differences across users."
        }
      ],
      baseContext()
    );

    expect(result.rejected).toEqual([]);
    expect(result.accepted).toHaveLength(1);
    expect(result.accepted[0]?.name).toBe("fe_income_age");
    expect(result.accepted[0]?.warnings).toEqual([]);
  });

  test("normalizes common AI response variants before validation", () => {
    const result = validateFeatureSuggestions(
      [
        {
          expression: "income / age",
          feature_name: "income_per_age",
          rationale: "Normalizes income by age.",
          expected_benefit: "Can reduce scale differences across users.",
          risk_notes: "Ratio is null when age is zero."
        },
        {
          expression: {
            operation: "square_root",
            input: "income"
          },
          name: "income_sqrt",
          reason: "Dampens skew in income.",
          benefit: "May improve linear model fit."
        },
        {
          transformation: {
            op: "extract_year",
            columnName: "signup_date"
          },
          name: "signup year",
          why: "Captures signup cohort.",
          expectedBenefit: "May reveal cohort-level effects.",
          risks: ["Year can overfit small samples."]
        },
        {
          operation: "product",
          columns: ["age", "income"],
          featureName: "age income product",
          reason: "Captures interaction between age and income.",
          expectedBenefit: "May expose multiplicative effects."
        }
      ],
      baseContext()
    );

    expect(result.rejected).toEqual([]);
    expect(result.accepted).toMatchObject([
      {
        expression: {
          op: "ratio",
          numerator: "income",
          denominator: "age"
        },
        name: "fe_income_per_age",
        warnings: ["Ratio is null when age is zero."]
      },
      {
        expression: { op: "sqrt", column: "income" },
        name: "fe_income_sqrt"
      },
      {
        expression: { op: "date_year", column: "signup_date" },
        name: "fe_signup_year",
        warnings: ["Year can overfit small samples."]
      },
      {
        expression: { op: "product", left: "age", right: "income" },
        name: "fe_age_income_product"
      }
    ]);
  });

  test("rejects target leakage and blocked columns", () => {
    const leakage = validateFeatureSuggestions(
      [
        {
          expression: { op: "difference", left: "age", right: "churn" },
          reason: "Uses the label.",
          expectedBenefit: "Would inflate performance."
        }
      ],
      baseContext({
        availableColumns: ["age", "churn"],
        numericColumns: ["age", "churn"],
        existingColumns: ["age", "churn"]
      })
    );

    expect(leakage.accepted).toEqual([]);
    expect(leakage.rejected[0]?.reason).toContain("target column");

    const blocked = validateFeatureSuggestions(
      [
        {
          expression: { op: "ratio", numerator: "income", denominator: "age" },
          reason: "Uses dropped data.",
          expectedBenefit: "None."
        }
      ],
      baseContext({ blockedColumns: ["income"] })
    );

    expect(blocked.accepted).toEqual([]);
    expect(blocked.rejected[0]?.reason).toContain("blocked column");
  });

  test("rejects duplicate and existing names", () => {
    const result = validateFeatureSuggestions(
      [
        {
          expression: { op: "square", column: "age" },
          name: "age_squared",
          reason: "Captures non-linearity.",
          expectedBenefit: "May improve fit."
        },
        {
          expression: { op: "product", left: "age", right: "income" },
          name: "age_squared",
          reason: "Second copy of same name.",
          expectedBenefit: "Should fail due to duplicate name."
        },
        {
          expression: { op: "date_year", column: "signup_date" },
          name: "signup_year",
          reason: "Collides with existing column.",
          expectedBenefit: "Should fail due to existing name."
        }
      ],
      baseContext({
        existingColumns: [
          "age",
          "income",
          "signup_date",
          "churn",
          "fe_signup_year"
        ]
      })
    );

    expect(result.accepted).toHaveLength(1);
    expect(result.rejected).toHaveLength(2);
    expect(result.rejected[0]?.reason).toContain("duplicated");
    expect(result.rejected[1]?.reason).toContain("already exists");
  });
});

describe("applyValidatedFeatureSuggestions", () => {
  test("adds computed values using validated names", () => {
    const output = applyValidatedFeatureSuggestions(
      [{ age: 20, income: 1000 }],
      [
        {
          expression: { op: "ratio", numerator: "income", denominator: "age" },
          name: "fe_income_to_age",
          reason: "Ratio",
          expectedBenefit: "Scale normalization",
          warnings: []
        }
      ]
    );

    expect(output).toEqual([{ age: 20, income: 1000, fe_income_to_age: 50 }]);
  });
});
