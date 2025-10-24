# Enhanced Screencast Script: AI-Assisted Development of Currency Profiling Module

## Preparation Notes
- **Total Duration**: ~15-20 minutes
- **Target Audience**: Developers with basic understanding of Python and data analysis (Vietnamese audience with English as second language)
- **Demonstration Environment**: PyCharm + Obsidian
- **Required Files**: Structure of PAMOLA.CORE project, sample CSV with currency data

## Step 1: Introduction Slide (00:00-01:00)
**[SLIDE: Title: "AI-Assisted Development of Currency Profiling Module in PAMOLA.CORE/PAMOLA"]**

**Narration:**
> Hello everyone! Today we will see how to develop a currency profiling module using AI assistance. We will work with the PAMOLA.CORE project, which is a prototype for PAMOLA CORE - a library designed for data anonymization and privacy assessment. Our goal is to create a complete module that analyzes currency fields in data tables and produces three types of outputs: CSV samples, JSON metrics, and PNG visualizations.

## Step 2: Project Environment Overview (01:00-02:30)
**[SCREEN: PyCharm with project structure and Obsidian]**

**Narration:**
> Here is our development environment. I'm using PyCharm to edit code and Obsidian for documentation. Let me show you the structure of the PAMOLA.CORE project. The main package is "core" which contains several sub-packages like "anonymization", "fake_data", "metrics", "privacy_models", and "profiling". 

> The "profiling" package has analyzers for different data types. We already have modules for numeric data, text data, and others. Today we will add a new module for currency data.

> Common utilities are in "pamola_core.utils" - these include modules for data input/output, visualization, logging, and progress tracking.

## Step 3: Creating Detailed Specification (02:30-04:30)
**[SCREEN: Creating specification document in Obsidian]**

**Narration:**
> First, we need to create a clear specification for our module. A good specification helps the AI understand exactly what we need. Let's create this document with detailed requirements.

> I'm writing the specification in English, describing the module purpose, technical architecture, input requirements, processing requirements, and output artifacts. This detailed specification will guide our AI assistant.

**[Show typing or pasting the specification]**

> Notice that I'm being very specific about where the module should be placed, what base classes to extend, what inputs it should accept, and what outputs it should produce. This level of detail is important for effective AI assistance.

## Step 4: AI Clarification Questions (04:30-06:30)
**[SCREEN: New chat with AI assistant, uploading specification]**

**Narration:**
> Now we'll start a new chat with the AI assistant. I'll upload our specification and ask the AI to analyze it and ask clarification questions before generating any code.

**[Show prompt to AI]:**
```
Based on the attached specification for the currency profiling module, please analyze the requirements and ask 5-7 clarification questions before starting code generation. Focus on technical implementation details, edge cases, and integration with existing system components.
```

**[Show AI response with questions]**

> The AI has asked some important questions about currency handling, input formats, mixed currencies, performance requirements, visualization details, and error handling. This helps identify gaps in our specification that we need to address.

**[Show typing answers to questions]**

> I'm providing answers to each question. This dialogue with the AI helps improve our specification and ensures the generated code will meet our needs.

## Step 5: Initial Code Generation (06:30-08:30)
**[SCREEN: PyCharm, creating new file]**

**Narration:**
> Now that we have clarified the requirements, let's generate the first version of our module. I'll create a new file at `pamola_core/profiling/analyzers/currency.py` and ask the AI to generate the implementation.

**[Show prompt to AI]:**
```
Now that we've clarified the requirements, please generate the implementation for `currency.py` module. Focus on the main operation class and pamola core functionality. We'll implement helper functions separately.
```

**[Show code generation result]**

> The AI has generated the initial version of our currency profiling module. Let's review the code structure. We have a `CurrencyProfileOperation` class that extends `FieldOperation` from the base classes. The code includes methods for analyzing currency values, calculating statistics, generating visualizations, and saving results.

## Step 6: Code Refactoring and Optimization (08:30-10:00)
**[SCREEN: PyCharm, reviewing generated code]**

**Narration:**
> The initial code is functional, but we can improve it. Let's ask the AI to suggest optimizations for performance, better code organization, improved error handling, and more robust currency detection.

**[Show prompt to AI]:**
```
Please analyze the generated code and suggest improvements in terms of:
1. Performance optimization for large datasets
2. Better separation of concerns (extracting utility methods)
3. Improved error handling
4. More robust currency detection logic
```

**[Show AI suggestions]**

> The AI has suggested several improvements. For example, extracting the currency detection logic to a separate utility function, using chunked processing for large datasets, adding more comprehensive error handling, and improving the statistics calculation.

**[Show implementing improvements]**

> I'm implementing these suggestions to improve our code. This step is important because the initial AI-generated code, while functional, often benefits from optimization and refactoring.

## Step 7: Debugging and Error Fixing (10:00-12:00)
**[SCREEN: PyCharm, running the module with test data]**

**Narration:**
> Now let's test our module with some sample data. I've prepared a CSV file with various currency formats and edge cases.

**[Show running the code and encountering an error]**

> We've encountered an error when parsing currency values with symbols. The error message shows it can't convert a value like '$1,234.56' to a numeric value. Let's ask the AI to help us fix this issue.

**[Show prompt to AI]:**
```
We're encountering an error when parsing currency values with symbols. The error occurs at row 23 with value '$1,234.56'. Please suggest a fix for the currency parsing logic to handle this common format.
```

**[Show AI solution and implementing it]**

> The AI suggests that we need to implement a more robust currency parsing function in our utils module. This function will remove currency symbols, handle different thousands separators, and convert the result to a numeric value.

**[Show creating/modifying currency_utils.py]**

> Now I'm implementing the suggested utility function in `pamola_core/profiling/commons/currency_utils.py`. This demonstrates a common scenario in AI-assisted development - identifying and fixing issues that arise during testing.

**[Show successful execution after fix]**

> After implementing the fix, our module now runs successfully! This shows the iterative process of development with AI assistance.

## Step 8: Analyzing Results (12:00-13:30)
**[SCREEN: Output artifacts - JSON, CSV, and PNG files]**

**Narration:**
> Now that our module is running correctly, let's examine the output artifacts it produces. As specified, we have three types of outputs.

**[Show metrics_currency.json]**

> First, the JSON metrics file contains statistical information about the currency field: minimum, maximum, average values, number of null values, detected currency type, and quartile boundaries.

**[Show sample_currency.csv]**

> Second, the CSV sample file contains representative examples from the dataset, including the ID field and the currency values.

**[Show distribution_currency.png and boxplot_currency.png]**

> Third, the visualization files include a histogram showing the distribution of values and a boxplot highlighting outliers. Note how the currency symbol is correctly displayed in the axis labels.

## Step 9: Creating Unit Tests (13:30-15:00)
**[SCREEN: PyCharm, creating test file]**

**Narration:**
> Good software development includes testing. Let's create unit tests for our module to ensure it works correctly in different scenarios.

**[Show prompt to AI]:**
```
Please generate comprehensive unit tests for our currency profiling module. Tests should cover:
1. Basic functionality with different input formats
2. Edge cases (empty dataset, all nulls, mixed formats)
3. Performance test for large datasets
4. Validation of all output artifacts
```

**[Show generated tests]**

> The AI has generated tests that check different aspects of our module. These tests ensure that the module handles various currency formats, properly processes edge cases, performs efficiently with large datasets, and generates all required output artifacts.

**[Show saving tests to tests/test_currency_profile.py]**

> I'm saving these tests to our test directory. In a real project, we would run these tests regularly to ensure our module continues to work correctly as we make changes.

## Step 10: Creating Documentation (15:00-17:00)
**[SCREEN: Obsidian, creating documentation file]**

**Narration:**
> The final step is to create documentation for our module. Good documentation makes it easier for other developers to use our code.

**[Show prompt to AI]:**
```
Generate comprehensive documentation for the currency profiling module in Markdown format for inclusion in our MkDocs system. Include:
1. Module purpose and architecture
2. Class diagram (using Mermaid)
3. Data flow diagram (using Mermaid)
4. Parameter reference table
5. Output artifact examples
6. Usage examples with code snippets
```

**[Show generated documentation]**

> The AI has created detailed documentation that explains what our module does, how it's structured, how data flows through it, what parameters it accepts, what outputs it produces, and how to use it in code.

**[Show diagrams and tables in documentation]**

> Notice the helpful diagrams that visualize the module's architecture and data flow. These make it easier to understand how the module works.

## Step 11: Integration Testing (17:00-18:00)
**[SCREEN: PyCharm, creating integration test script]**

**Narration:**
> Let's also test how our module works as part of a complete workflow in the PAMOLA.CORE/PAMOLA system. We'll create a test script that loads data, profiles currency fields, and generates a report.

**[Show running integration test]**

> Our module integrates successfully with the rest of the system! The complete workflow runs without errors and produces the expected results.

## Step 12: Comparison with Traditional Development (18:00-19:00)
**[SLIDE: Table comparing AI-assisted vs traditional development times]**

**Narration:**
> Let's compare how long it took us to develop this module using AI assistance versus how long it would take using traditional methods.

> With AI assistance, we completed the specification in 15 minutes, coding in 30 minutes, debugging in 20 minutes, testing in 15 minutes, and documentation in 15 minutes - a total of about 1.5 hours.

> Traditional development would typically take about 10 hours - 1-2 hours for specification, 3-4 hours for coding, 1-2 hours for debugging, 1-2 hours for testing, and 2-3 hours for documentation.

> This represents a 6-7x improvement in development efficiency!

## Step 13: Conclusion (19:00-20:00)
**[SLIDE: Key Takeaways]**

**Narration:**
> To summarize what we've learned:

> We completed a full development cycle from concept to documented, tested code using AI assistance.

> The quality of our specification was crucial for effective AI assistance.

> The best results came from combining AI automation with human expertise.

> This approach can be applied to developing other modules in the PAMOLA.CORE/PAMOLA system.

> There are some limitations: we still needed to validate the code, understand the system architecture, and fix errors the AI couldn't handle.

> Thank you for watching this demonstration of AI-assisted development!