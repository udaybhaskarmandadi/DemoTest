# Databricks notebook source_test_to ccheck_commit
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # PySpark CASE WHEN — Databricks Training Material
# MAGIC
# MAGIC ## `when()` and `otherwise()` in PySpark
# MAGIC
# MAGIC This notebook is designed as instructor-led training material for Databricks.
# MAGIC
# MAGIC ### Topics covered
# MAGIC - Understanding conditional transformations
# MAGIC - `when()` and `otherwise()`
# MAGIC - Multiple conditions
# MAGIC - AND / OR conditions
# MAGIC - Comparisons and ranges
# MAGIC - `isin()`
# MAGIC - NULL handling
# MAGIC - Creating derived columns
# MAGIC - Conditional numeric calculations
# MAGIC - SQL `CASE WHEN` vs PySpark
# MAGIC - Multiple conditional columns
# MAGIC - Real-world employee and customer examples
# MAGIC - Practical ETL patterns
# MAGIC - Common mistakes and best practices
# MAGIC
# MAGIC > Core pattern:
# MAGIC >
# MAGIC > `F.when(condition, value).otherwise(default_value)`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is CASE WHEN?
# MAGIC
# MAGIC Conditional logic is required frequently in data engineering.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC - Salary >= 100000 → High
# MAGIC - Salary >= 70000 → Medium
# MAGIC - Otherwise → Low
# MAGIC
# MAGIC In SQL, this is written using `CASE WHEN`.
# MAGIC
# MAGIC In PySpark, the equivalent is:
# MAGIC
# MAGIC ```python
# MAGIC F.when(condition, value).otherwise(default_value)
# MAGIC ```
# MAGIC
# MAGIC Think of it as:
# MAGIC
# MAGIC ```text
# MAGIC IF condition THEN value
# MAGIC ELSE default_value
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import PySpark Functions
# MAGIC
# MAGIC We will use `pyspark.sql.functions` throughout the notebook.
# MAGIC
# MAGIC Using the alias `F` makes PySpark transformations shorter and easier to read.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create a Sample DataFrame
# MAGIC
# MAGIC We will start with employee data containing:
# MAGIC
# MAGIC - employee ID
# MAGIC - employee name
# MAGIC - department
# MAGIC - salary
# MAGIC - years of experience

# COMMAND ----------

data = [
    (101, "Ravi",   "IT",      120000, 8),
    (102, "Rahul",  "HR",       65000, 4),
    (103, "Ram",    "Finance",  85000, 6),
    (104, "John",   "IT",       95000, 5),
    (105, "David",  "HR",       40000, 2),
    (106, "Rajesh", None,        None, 1)
]

df = spark.createDataFrame(
    data,
    ["emp_id", "name", "department", "salary", "experience"]
)

display(df)

# COMMAND ----------

df.createOrReplaceTempView("tablecase")

# COMMAND ----------

# MAGIC %sql
# MAGIC select *,case when salary < 50000 then 'High' 
# MAGIC               when salary > 70000 and salary < 95000 then 'medium' else 'Low' end as salary_status from tablecase

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect the Schema

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Basic `when()`
# MAGIC
# MAGIC Suppose we want to classify employees as:
# MAGIC
# MAGIC - Pass when salary is at least 70000
# MAGIC - No value otherwise
# MAGIC
# MAGIC `when()` evaluates the condition and returns the specified value when the condition is true.

# COMMAND ----------

df_basic = df.withColumn(
    "salary_status",
    F.when(F.col("salary") >= 70000, "Qualified")
)

display(df_basic)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Important behavior
# MAGIC
# MAGIC If no `otherwise()` is supplied, rows that do not satisfy the condition receive `NULL`.
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```text
# MAGIC salary = 120000 → Qualified
# MAGIC salary = 65000  → NULL
# MAGIC salary = 40000  → NULL
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. `when()` with `otherwise()`
# MAGIC
# MAGIC `otherwise()` provides the default value when none of the preceding conditions match.

# COMMAND ----------

df_status = df.withColumn(
    "salary_status",
    F.when(F.col("salary") >= 70000, "Qualified")
     .when((F.col("salary") < 70000) & (F.col("salary") >= 50000), "Less Qualified")
     .otherwise("Not Qualified")
)

display(df_status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Multiple `when()` Conditions
# MAGIC
# MAGIC Multiple conditions can be chained.
# MAGIC
# MAGIC Evaluation occurs from top to bottom.
# MAGIC
# MAGIC Example salary bands:
# MAGIC
# MAGIC ```text
# MAGIC >= 100000  → High
# MAGIC >= 70000   → Medium
# MAGIC < 70000    → Low
# MAGIC ```
# MAGIC
# MAGIC The first matching condition is used.

# COMMAND ----------

df_salary_band = df.withColumn(
    "salary_band",
    F.when(F.col("salary") >= 100000, "High")
     .when(F.col("salary") >= 70000, "Medium")
     .otherwise("Low")
)

display(df_salary_band)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Why condition order matters
# MAGIC
# MAGIC Consider:
# MAGIC
# MAGIC ```python
# MAGIC F.when(F.col("salary") >= 70000, "Medium")
# MAGIC  .when(F.col("salary") >= 100000, "High")
# MAGIC ```
# MAGIC
# MAGIC A salary of 120000 matches the first condition, so it will be classified as `Medium`.
# MAGIC
# MAGIC Therefore, when conditions overlap, put the more specific / higher-priority condition first.

# COMMAND ----------

# Correct ordering
df_salary_band = df.withColumn(
    "salary_band",
    F.when(F.col("salary") >= 100000, "High")
     .when(F.col("salary") >= 70000, "Medium")
     .otherwise("Low")
)

display(df_salary_band)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Multiple Conditions with AND
# MAGIC
# MAGIC Use `&` for logical AND.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC An employee is eligible when:
# MAGIC
# MAGIC - salary >= 80000
# MAGIC - AND department is IT

# COMMAND ----------

df_bonus = df.withColumn(
    "bonus_eligible",
    F.when(
        (F.col("salary") >= 80000) &
        (F.col("department") == "IT"),
        "Yes"
    ).otherwise("No")
)

display(df_bonus)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Syntax rule
# MAGIC
# MAGIC Put each comparison inside parentheses:
# MAGIC
# MAGIC ```python
# MAGIC (condition1) & (condition2)
# MAGIC ```
# MAGIC
# MAGIC Avoid writing Python's `and` for Spark Column expressions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Multiple Conditions with OR
# MAGIC
# MAGIC Use `|` for logical OR.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC An employee is considered a priority employee if:
# MAGIC
# MAGIC - salary >= 100000
# MAGIC - OR experience >= 8

# COMMAND ----------

df_priority = df.withColumn(
    "priority_employee",
    F.when(
        (F.col("salary") >= 100000) |
        (F.col("experience") >= 8),
        "Yes"
    ).otherwise("No")
)

display(df_priority)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Combining AND and OR
# MAGIC
# MAGIC Parentheses make the business rule explicit.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC salary >= 80000 AND department = IT
# MAGIC OR
# MAGIC experience >= 10
# MAGIC ```

# COMMAND ----------

df_complex = df.withColumn(
    "special_category",
    F.when(((F.col("salary") >= 80000) &
            (F.col("department") == "IT")
        ) |
        (F.col("experience") >= 10),
        "Special"
    ).otherwise("Standard")
)

display(df_complex)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Comparing String Columns
# MAGIC
# MAGIC Conditional logic is not limited to numeric columns.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC - IT → Technical
# MAGIC - Finance → Finance
# MAGIC - HR → Human Resources
# MAGIC - anything else → Other

# COMMAND ----------

df_department = df.withColumn(
    "department_group",
    F.when(F.col("department") == "IT", "Technical")
     .when(F.col("department") == "Finance", "Finance")
     .when(F.col("department") == "HR", "Human Resources")
     .otherwise("Other")
)

display(df_department)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Using `isin()`
# MAGIC
# MAGIC When several values should receive the same result, `isin()` is cleaner than writing multiple OR conditions.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC ```text
# MAGIC IT or Finance → Business Critical
# MAGIC HR            → Corporate
# MAGIC ```

# COMMAND ----------

df_business = df.withColumn(
    "business_group",
    F.when(
        F.col("department").isin("IT", "Finance"),
        "Business Critical"
    )
    .when(
        F.col("department") == "HR",
        "Corporate"
    )
    .otherwise("Unknown")
)

display(df_business)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Handling NULL Values
# MAGIC
# MAGIC NULL is common in real-world source data.
# MAGIC
# MAGIC Use:
# MAGIC
# MAGIC ```python
# MAGIC F.col("column").isNull()
# MAGIC ```
# MAGIC
# MAGIC and:
# MAGIC
# MAGIC ```python
# MAGIC F.col("column").isNotNull()
# MAGIC ```
# MAGIC
# MAGIC Do not use:
# MAGIC
# MAGIC ```python
# MAGIC F.col("column") == None
# MAGIC ```
# MAGIC
# MAGIC for Spark NULL checks.

# COMMAND ----------

df_null = df.withColumn(
    "data_status",
    F.when(F.col("department").isNull(), "Department Missing")
     .otherwise("Department Available")
)

display(df_null)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. NULL Handling with Multiple Conditions
# MAGIC
# MAGIC A useful pattern is to handle NULL first and then evaluate business rules.

# COMMAND ----------

df_salary_status = df.withColumn(
    "salary_category",
    F.when(F.col("salary").isNull(), "Salary Missing")
     .when(F.col("salary") >= 100000, "High")
     .when(F.col("salary") >= 70000, "Medium")
     .otherwise("Low")
)

display(df_salary_status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Conditional Numeric Values
# MAGIC
# MAGIC `when()` can return numbers, not only strings.
# MAGIC
# MAGIC Example bonus amounts:
# MAGIC
# MAGIC ```text
# MAGIC salary >= 100000 → 10000
# MAGIC salary >= 70000  → 5000
# MAGIC otherwise        → 1000
# MAGIC ```

# COMMAND ----------

df_bonus_amount = df.withColumn(
    "bonus_amount",
    F.when(F.col("salary").isNull(), 0)
     .when(F.col("salary") >= 100000, 10000)
     .when(F.col("salary") >= 70000, 5000)
     .otherwise(1000)
)

display(df_bonus_amount)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Conditional Calculation Using Existing Columns
# MAGIC
# MAGIC The value returned by `when()` can be a calculation.
# MAGIC
# MAGIC For example, calculate a bonus as a percentage of salary.

# COMMAND ----------

df_bonus_calc = df.withColumn(
    "bonus",
    F.when(
        F.col("salary") >= 100000,
        F.col("salary") * 0.10
    )
    .when(
        F.col("salary") >= 70000,
        F.col("salary") * 0.05
    )
    .otherwise(0)
)

display(df_bonus_calc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. SQL CASE WHEN Equivalent
# MAGIC
# MAGIC ### SQL
# MAGIC
# MAGIC ```sql
# MAGIC CASE
# MAGIC     WHEN salary >= 100000 THEN 'High'
# MAGIC     WHEN salary >= 70000 THEN 'Medium'
# MAGIC     ELSE 'Low'
# MAGIC END
# MAGIC ```
# MAGIC
# MAGIC ### PySpark
# MAGIC
# MAGIC ```python
# MAGIC F.when(F.col("salary") >= 100000, "High")  .when(F.col("salary") >= 70000, "Medium")  .otherwise("Low")
# MAGIC ```
# MAGIC
# MAGIC The business logic is the same.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Using `select()` Instead of `withColumn()`
# MAGIC
# MAGIC Conditional expressions can be created directly inside `select()`.

# COMMAND ----------

result = df.select(
    "emp_id",
    "name",
    "salary",
    F.when(F.col("salary") >= 100000, "High")
     .when(F.col("salary") >= 70000, "Medium")
     .otherwise("Low")
     .alias("salary_band")
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Creating Multiple Derived Columns
# MAGIC
# MAGIC In a real ETL pipeline, several business rules may be applied to the same DataFrame.
# MAGIC
# MAGIC Here we create:
# MAGIC
# MAGIC - salary band
# MAGIC - bonus eligibility
# MAGIC - department group
# MAGIC - employee level

# COMMAND ----------

result = (
    df
    .withColumn(
        "salary_band",
        F.when(F.col("salary").isNull(), "Unknown")
         .when(F.col("salary") >= 100000, "High")
         .when(F.col("salary") >= 70000, "Medium")
         .otherwise("Low")
    )
    .withColumn(
        "bonus_eligible",
        F.when(
            (F.col("salary") >= 80000) &
            (F.col("department") == "IT"),
            "Yes"
        ).otherwise("No")
    )
    .withColumn(
        "department_group",
        F.when(
            F.col("department").isin("IT", "Finance"),
            "Business"
        )
        .when(
            F.col("department") == "HR",
            "Corporate"
        )
        .otherwise("Unknown")
    )
    .withColumn(
        "employee_level",
        F.when(F.col("experience") >= 8, "Senior")
         .when(F.col("experience") >= 5, "Mid-Level")
         .otherwise("Junior")
    )
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # 20. Real-World Customer Example
# MAGIC
# MAGIC Conditional logic is commonly used when standardizing customer data.
# MAGIC
# MAGIC Example business rules:
# MAGIC
# MAGIC - Age >= 60 → Senior
# MAGIC - Age >= 30 → Adult
# MAGIC - Age < 30 → Young
# MAGIC - NULL → Unknown

# COMMAND ----------

customer_data = [
    (1, "Ravi", 65, "India", 120000),
    (2, "Rahul", 42, "India", 80000),
    (3, "John", 25, "USA", 70000),
    (4, "David", 18, "UK", 30000),
    (5, "Rajesh", None, None, None)
]

customers = spark.createDataFrame(
    customer_data,
    ["customer_id", "name", "age", "country", "income"]
)

display(customers)

# COMMAND ----------

customers_enriched = (
    customers
    .withColumn(
        "age_group",
        F.when(F.col("age").isNull(), "Unknown")
         .when(F.col("age") >= 60, "Senior")
         .when(F.col("age") >= 30, "Adult")
         .otherwise("Young")
    )
    .withColumn(
        "region",
        F.when(
            F.col("country").isin("India", "Singapore", "Japan"),
            "APAC"
        )
        .when(
            F.col("country").isin("USA", "Canada"),
            "North America"
        )
        .when(
            F.col("country").isin("UK", "Germany", "France"),
            "Europe"
        )
        .otherwise("Other")
    )
)

display(customers_enriched)

# COMMAND ----------

# MAGIC %md
# MAGIC # 21. Conditional Data Quality Flag
# MAGIC
# MAGIC `when()` is very useful for data-quality rules.
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC A customer record is valid only when:
# MAGIC
# MAGIC - name is not NULL
# MAGIC - country is not NULL
# MAGIC - income is not NULL

# COMMAND ----------

customers_quality = customers.withColumn(
    "data_quality",
    F.when(
        F.col("name").isNull() |
        F.col("country").isNull() |
        F.col("income").isNull(),
        "Invalid"
    ).otherwise("Valid")
)

display(customers_quality)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Creating a Reusable Data Quality Status
# MAGIC
# MAGIC Multiple conditions can identify different problems.
# MAGIC
# MAGIC Priority order:
# MAGIC
# MAGIC 1. Missing name
# MAGIC 2. Missing country
# MAGIC 3. Missing income
# MAGIC 4. Valid

# COMMAND ----------

customers_quality_detail = customers.withColumn(
    "quality_status",
    F.when(F.col("name").isNull(), "Missing Name")
     .when(F.col("country").isNull(), "Missing Country")
     .when(F.col("income").isNull(), "Missing Income")
     .otherwise("Valid")
)

display(customers_quality_detail)

# COMMAND ----------

# MAGIC %md
# MAGIC # 23. Real ETL Pattern
# MAGIC
# MAGIC A common Databricks transformation pipeline looks like:
# MAGIC
# MAGIC ```text
# MAGIC Source
# MAGIC   ↓
# MAGIC Read Data
# MAGIC   ↓
# MAGIC Clean / Standardize
# MAGIC   ↓
# MAGIC when() / otherwise()
# MAGIC   ↓
# MAGIC Derived Business Columns
# MAGIC   ↓
# MAGIC Filter
# MAGIC   ↓
# MAGIC Write Silver / Gold
# MAGIC ```
# MAGIC
# MAGIC Example business transformation:
# MAGIC
# MAGIC ```text
# MAGIC Raw department
# MAGIC       ↓
# MAGIC CASE WHEN
# MAGIC       ↓
# MAGIC Standardized department group
# MAGIC ```
# MAGIC
# MAGIC This is one of the most common patterns used in Silver-layer transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC # 24. Combining `when()` with Other Functions
# MAGIC
# MAGIC Conditional logic can be combined with functions such as:
# MAGIC
# MAGIC - `upper()`
# MAGIC - `lower()`
# MAGIC - `trim()`
# MAGIC - `coalesce()`
# MAGIC - `regexp_replace()`
# MAGIC - `datediff()`
# MAGIC - `to_date()`
# MAGIC - `round()`
# MAGIC - `lit()`

# COMMAND ----------

data = [
    (1, "  ravi ", "it"),
    (2, "RAHUL", "hr"),
    (3, " Ram ", "finance")
]

employees_clean = spark.createDataFrame(
    data,
    ["id", "name", "department"]
)

result = employees_clean.withColumn(
    "standard_department",
    F.when(
        F.upper(F.trim(F.col("department"))) == "IT",
        "Information Technology"
    )
    .when(
        F.upper(F.trim(F.col("department"))) == "HR",
        "Human Resources"
    )
    .otherwise(
        F.initcap(F.trim(F.col("department")))
    )
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # 25. Using `coalesce()` with Conditional Logic
# MAGIC
# MAGIC Sometimes a source field may be NULL and a default value is required.
# MAGIC
# MAGIC `coalesce()` returns the first non-NULL expression.

# COMMAND ----------

result = df.withColumn(
    "department_final",
    F.coalesce(
        F.col("department"),
        F.lit("Unknown")
    )
)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC # 26. Common Mistakes
# MAGIC
# MAGIC ### Mistake 1 — Using Python `and`
# MAGIC
# MAGIC Incorrect:
# MAGIC
# MAGIC ```python
# MAGIC F.col("salary") >= 80000 and F.col("department") == "IT"
# MAGIC ```
# MAGIC
# MAGIC Correct:
# MAGIC
# MAGIC ```python
# MAGIC (F.col("salary") >= 80000) &
# MAGIC (F.col("department") == "IT")
# MAGIC ```
# MAGIC
# MAGIC ### Mistake 2 — Using Python `or`
# MAGIC
# MAGIC Incorrect:
# MAGIC
# MAGIC ```python
# MAGIC condition1 or condition2
# MAGIC ```
# MAGIC
# MAGIC Correct:
# MAGIC
# MAGIC ```python
# MAGIC condition1 | condition2
# MAGIC ```
# MAGIC
# MAGIC ### Mistake 3 — Incorrect NULL comparison
# MAGIC
# MAGIC Avoid:
# MAGIC
# MAGIC ```python
# MAGIC F.col("salary") == None
# MAGIC ```
# MAGIC
# MAGIC Use:
# MAGIC
# MAGIC ```python
# MAGIC F.col("salary").isNull()
# MAGIC ```
# MAGIC
# MAGIC ### Mistake 4 — Forgetting `otherwise()`
# MAGIC
# MAGIC Without `otherwise()`, unmatched rows become NULL.

# COMMAND ----------

# MAGIC %md
# MAGIC # 27. Best Practices
# MAGIC
# MAGIC ### 1. Order overlapping conditions carefully
# MAGIC
# MAGIC Put higher-priority conditions first.
# MAGIC
# MAGIC ### 2. Handle NULL explicitly
# MAGIC
# MAGIC Especially for business-critical columns.
# MAGIC
# MAGIC ### 3. Use `isin()` for lists of values
# MAGIC
# MAGIC It is cleaner than many OR conditions.
# MAGIC
# MAGIC ### 4. Use meaningful column names
# MAGIC
# MAGIC For example:
# MAGIC
# MAGIC ```text
# MAGIC salary_band
# MAGIC bonus_eligible
# MAGIC data_quality
# MAGIC customer_segment
# MAGIC ```
# MAGIC
# MAGIC ### 5. Keep complex business rules readable
# MAGIC
# MAGIC Break very large transformations into logical stages when necessary.
# MAGIC
# MAGIC ### 6. Validate the output
# MAGIC
# MAGIC Use:
# MAGIC
# MAGIC ```python
# MAGIC display(df)
# MAGIC df.printSchema()
# MAGIC df.groupBy(...).count()
# MAGIC ```
# MAGIC
# MAGIC to verify the transformation.

# COMMAND ----------

# MAGIC %md
# MAGIC # 28. Complete End-to-End Example
# MAGIC
# MAGIC This example combines the main concepts into a realistic employee transformation.
# MAGIC
# MAGIC Business rules:
# MAGIC
# MAGIC - Missing salary → Unknown salary band
# MAGIC - >= 100000 → High
# MAGIC - >= 70000 → Medium
# MAGIC - otherwise → Low
# MAGIC - IT + salary >= 80000 → Bonus Eligible
# MAGIC - IT / Finance → Business
# MAGIC - HR → Corporate
# MAGIC - 8+ years → Senior
# MAGIC - 5+ years → Mid-Level
# MAGIC - otherwise → Junior

# COMMAND ----------

from pyspark.sql import functions as F

data = [
    (101, "Ravi",   "IT",      120000, 8),
    (102, "Rahul",  "HR",       65000, 4),
    (103, "Ram",    "Finance",  85000, 6),
    (104, "John",   "IT",       95000, 5),
    (105, "David",  "HR",       40000, 2),
    (106, "Rajesh", None,        None, 1)
]

df = spark.createDataFrame(
    data,
    ["emp_id", "name", "department", "salary", "experience"]
)

final_df = (
    df
    .withColumn(
        "salary_band",
        F.when(F.col("salary").isNull(), "Unknown")
         .when(F.col("salary") >= 100000, "High")
         .when(F.col("salary") >= 70000, "Medium")
         .otherwise("Low")
    )
    .withColumn(
        "bonus_eligible",
        F.when(
            (F.col("salary") >= 80000) &
            (F.col("department") == "IT"),
            "Yes"
        ).otherwise("No")
    )
    .withColumn(
        "department_group",
        F.when(
            F.col("department").isin("IT", "Finance"),
            "Business"
        )
        .when(
            F.col("department") == "HR",
            "Corporate"
        )
        .otherwise("Unknown")
    )
    .withColumn(
        "employee_level",
        F.when(F.col("experience") >= 8, "Senior")
         .when(F.col("experience") >= 5, "Mid-Level")
         .otherwise("Junior")
    )
)

display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 29. Key Takeaways
# MAGIC
# MAGIC ## Core PySpark pattern
# MAGIC
# MAGIC ```python
# MAGIC F.when(condition, value)
# MAGIC  .when(condition, value)
# MAGIC  .otherwise(default_value)
# MAGIC ```
# MAGIC
# MAGIC ### Operators
# MAGIC
# MAGIC ```text
# MAGIC AND       → &
# MAGIC OR        → |
# MAGIC Equal     → ==
# MAGIC Not equal → !=
# MAGIC IN        → isin()
# MAGIC NULL      → isNull()
# MAGIC NOT NULL  → isNotNull()
# MAGIC ```
# MAGIC
# MAGIC ### Remember
# MAGIC
# MAGIC ```text
# MAGIC SQL CASE WHEN
# MAGIC        ↓
# MAGIC PySpark when()
# MAGIC        ↓
# MAGIC PySpark otherwise()
# MAGIC ```
# MAGIC
# MAGIC `when()` / `otherwise()` is primarily used to build conditional expressions and derived columns in PySpark DataFrame transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC # 30. Trainer Notes
# MAGIC
# MAGIC ### Recommended teaching sequence
# MAGIC
# MAGIC 1. Explain the SQL `CASE WHEN` concept.
# MAGIC 2. Show the equivalent PySpark syntax.
# MAGIC 3. Start with one condition.
# MAGIC 4. Add `otherwise()`.
# MAGIC 5. Add multiple conditions.
# MAGIC 6. Demonstrate condition ordering.
# MAGIC 7. Introduce `&` and `|`.
# MAGIC 8. Introduce `isin()`.
# MAGIC 9. Explain NULL handling.
# MAGIC 10. Show numeric calculations.
# MAGIC 11. Apply the logic to a realistic ETL scenario.
# MAGIC 12. Review common mistakes.
# MAGIC 13. Run the complete end-to-end example.
# MAGIC
# MAGIC ### Suggested hands-on practice
# MAGIC
# MAGIC After completing the explanation, have the candidate modify the examples themselves. The notebook is intentionally structured so the concepts are demonstrated first and practice can be added without turning the lesson into a question-and-answer document.
