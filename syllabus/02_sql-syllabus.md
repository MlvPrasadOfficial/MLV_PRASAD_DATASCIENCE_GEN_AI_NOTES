# SQL for Data Science - Comprehensive Syllabus

## 📚 Course Overview
This syllabus covers SQL from basic queries to advanced data analysis techniques, focusing on practical applications in data science and business intelligence.

---

## 🎯 Learning Objectives
- Master fundamental SQL syntax and operations
- Develop advanced querying and data manipulation skills
- Learn database design and optimization principles
- Apply SQL to real-world data analysis scenarios
- Integrate SQL with modern data science workflows

---

## 📖 Module 1: SQL Fundamentals (Weeks 1-2)

### 1.1 Database Concepts & Setup
- [ ] Relational database theory
- [ ] Database vs. Data warehouse concepts
- [ ] Setting up database environments (PostgreSQL, MySQL, SQLite)
- [ ] Database clients and tools (pgAdmin, MySQL Workbench, DBeaver)
- [ ] Understanding schemas and catalogs

### 1.2 Basic SQL Structure
- [ ] SQL syntax and conventions
- [ ] Data types and constraints
- [ ] NULL values and handling
- [ ] Comments and code organization
- [ ] Case sensitivity considerations

### 1.3 Data Retrieval Basics
- [ ] SELECT statements and column selection
- [ ] WHERE clause and filtering
- [ ] ORDER BY for sorting results
- [ ] LIMIT and OFFSET for pagination
- [ ] DISTINCT for unique values

### 1.4 Basic Operators
- [ ] Comparison operators (=, !=, <, >, <=, >=)
- [ ] Logical operators (AND, OR, NOT)
- [ ] Pattern matching (LIKE, ILIKE)
- [ ] Range operations (BETWEEN, IN, NOT IN)
- [ ] NULL checking (IS NULL, IS NOT NULL)

---

## 🔧 Module 2: Data Manipulation (Weeks 3-4)

### 2.1 String Functions
- [ ] String concatenation and manipulation
- [ ] UPPER, LOWER, TRIM, LENGTH functions
- [ ] SUBSTRING and string extraction
- [ ] REPLACE and TRANSLATE functions
- [ ] Regular expressions (REGEXP, SIMILAR TO)

### 2.2 Date and Time Functions
- [ ] Date arithmetic and intervals
- [ ] DATE_PART and EXTRACT functions
- [ ] Date formatting and conversion
- [ ] Timezone handling
- [ ] Working with timestamps

### 2.3 Numeric Functions
- [ ] Mathematical operations and functions
- [ ] ROUND, CEIL, FLOOR operations
- [ ] ABS, MOD, POWER functions
- [ ] RANDOM and statistical functions
- [ ] Type casting and conversion

### 2.4 Conditional Logic
- [ ] CASE statements and conditional expressions
- [ ] COALESCE and NULLIF functions
- [ ] IF statements (MySQL specific)
- [ ] Nested conditions and complex logic

---

## 📊 Module 3: Aggregation & Grouping (Week 5)

### 3.1 Aggregate Functions
- [ ] COUNT, SUM, AVG, MIN, MAX
- [ ] COUNT(DISTINCT) for unique counting
- [ ] Handling NULL values in aggregations
- [ ] Statistical functions (STDDEV, VARIANCE)

### 3.2 GROUP BY Operations
- [ ] Single and multiple column grouping
- [ ] HAVING clause for group filtering
- [ ] GROUP BY with expressions
- [ ] Grouping sets and rollup operations

### 3.3 Window Functions (Advanced)
- [ ] ROW_NUMBER, RANK, DENSE_RANK
- [ ] LEAD and LAG functions
- [ ] Cumulative aggregations
- [ ] Partitioning and ordering
- [ ] Frame specifications

---

## 🔗 Module 4: Table Relationships & Joins (Weeks 6-7)

### 4.1 Join Fundamentals
- [ ] INNER JOIN concepts and syntax
- [ ] LEFT/RIGHT OUTER JOINs
- [ ] FULL OUTER JOIN
- [ ] CROSS JOIN and Cartesian products
- [ ] Self-joins and recursive relationships

### 4.2 Advanced Join Techniques
- [ ] Multiple table joins
- [ ] Join conditions and performance
- [ ] Non-equi joins
- [ ] Anti-joins and semi-joins
- [ ] Joining on different data types

### 4.3 Set Operations
- [ ] UNION and UNION ALL
- [ ] INTERSECT operations
- [ ] EXCEPT/MINUS operations
- [ ] Combining set operations with joins

### 4.4 Subqueries
- [ ] Single-value subqueries
- [ ] Multi-value subqueries (IN, ANY, ALL)
- [ ] Correlated subqueries
- [ ] EXISTS and NOT EXISTS
- [ ] Subquery optimization techniques

---

## 🏗️ Module 5: Database Design & DDL (Week 8)

### 5.1 Table Creation & Management
- [ ] CREATE TABLE statements
- [ ] Data type selection strategies
- [ ] Constraints (PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK)
- [ ] ALTER TABLE operations
- [ ] DROP and TRUNCATE operations

### 5.2 Data Modification
- [ ] INSERT statements (single and bulk)
- [ ] UPDATE operations with conditions
- [ ] DELETE operations and safety practices
- [ ] UPSERT operations (INSERT ... ON CONFLICT)
- [ ] Batch processing strategies

### 5.3 Indexes & Performance
- [ ] Index types and creation
- [ ] Composite indexes
- [ ] Index maintenance and monitoring
- [ ] Query performance analysis
- [ ] Execution plan interpretation

---

## 📈 Module 6: Advanced Analytics (Weeks 9-10)

### 6.1 Common Table Expressions (CTEs)
- [ ] Simple CTE syntax and usage
- [ ] Recursive CTEs
- [ ] Multiple CTEs in single query
- [ ] CTE vs. subquery performance
- [ ] Temporary result set management

### 6.2 Advanced Window Functions
- [ ] Moving averages and rolling calculations
- [ ] Percentile and ranking functions
- [ ] First/last value functions
- [ ] Window frame specifications
- [ ] Analytical functions for time series

### 6.3 Pivot and Unpivot Operations
- [ ] Manual pivot with CASE statements
- [ ] Native PIVOT operations (SQL Server)
- [ ] Dynamic pivot techniques
- [ ] UNPIVOT for normalization
- [ ] Cross-tab reporting

### 6.4 Data Quality & Validation
- [ ] Duplicate detection strategies
- [ ] Data profiling queries
- [ ] Constraint validation
- [ ] Outlier detection
- [ ] Data consistency checks

---

## 🔍 Module 7: Query Optimization (Week 11)

### 7.1 Performance Fundamentals
- [ ] Query execution plans
- [ ] Index usage and optimization
- [ ] Join order and algorithms
- [ ] WHERE clause optimization
- [ ] Avoiding common performance pitfalls

### 7.2 Advanced Optimization
- [ ] Partitioning strategies
- [ ] Materialized views
- [ ] Query hints and forcing plans
- [ ] Statistics and cost estimation
- [ ] Parallel query execution

### 7.3 Monitoring & Troubleshooting
- [ ] Performance monitoring tools
- [ ] Query profiling techniques
- [ ] Lock detection and resolution
- [ ] Resource utilization analysis
- [ ] Slow query identification

---

## 🌐 Module 8: Modern SQL & Integration (Week 12)

### 8.1 JSON and Semi-Structured Data
- [ ] JSON data type and operations
- [ ] JSON path expressions
- [ ] Extracting and manipulating JSON
- [ ] Array operations in PostgreSQL
- [ ] NoSQL integration patterns

### 8.2 Advanced Data Types
- [ ] Array and composite types
- [ ] Geographic data (PostGIS basics)
- [ ] User-defined data types
- [ ] Large object handling
- [ ] Full-text search capabilities

### 8.3 Stored Procedures & Functions
- [ ] Function creation and syntax
- [ ] Parameter handling
- [ ] Control flow in stored procedures
- [ ] Exception handling
- [ ] Performance considerations

---

## 🔗 Module 9: SQL in Data Science Ecosystem (Week 13)

### 9.1 Database Connectivity
- [ ] Connecting from Python (SQLAlchemy, psycopg2)
- [ ] R database integration (DBI, RPostgreSQL)
- [ ] ODBC and JDBC connections
- [ ] Connection pooling and management
- [ ] Security and authentication

### 9.2 ETL and Data Pipelines
- [ ] Data extraction patterns
- [ ] Transformation workflows
- [ ] Loading strategies and bulk operations
- [ ] Change data capture (CDC)
- [ ] Data validation and error handling

### 9.3 Cloud and Big Data Integration
- [ ] BigQuery fundamentals
- [ ] Snowflake data warehouse
- [ ] Amazon Redshift basics
- [ ] Spark SQL introduction
- [ ] Data lake query patterns

---

## 📊 Module 10: Business Intelligence & Reporting (Week 14)

### 10.1 Analytical Queries
- [ ] Time series analysis patterns
- [ ] Cohort analysis techniques
- [ ] Funnel analysis queries
- [ ] A/B testing data analysis
- [ ] Customer segmentation SQL

### 10.2 KPI and Metrics Calculation
- [ ] Business metrics definitions
- [ ] Ratio and percentage calculations
- [ ] Year-over-year comparisons
- [ ] Moving averages and trends
- [ ] Forecasting with SQL

### 10.3 Reporting Best Practices
- [ ] Query organization and documentation
- [ ] Parameterized queries
- [ ] Report automation strategies
- [ ] Data freshness and accuracy
- [ ] Performance optimization for reports

---

## 🛠️ Practical Projects

### Project 1: E-commerce Analytics (Week 15)
- [ ] Design normalized database schema
- [ ] Implement customer behavior analysis
- [ ] Create sales performance dashboards
- [ ] Build recommendation query logic

### Project 2: Financial Data Analysis (Week 16)
- [ ] Time series financial data processing
- [ ] Risk analysis and portfolio queries
- [ ] Regulatory reporting automation
- [ ] Real-time data processing patterns

---

## 🗄️ Database Platforms & Tools

### Primary Platforms
- **PostgreSQL**: Advanced features, JSON, arrays
- **MySQL**: Web applications, performance tuning
- **SQLite**: Embedded applications, data analysis
- **SQL Server**: Enterprise features, T-SQL
- **Oracle**: Enterprise-grade, advanced analytics

### Analysis Tools
- **DBeaver**: Universal database client
- **pgAdmin**: PostgreSQL administration
- **DataGrip**: JetBrains database IDE
- **Tableau**: BI and visualization
- **Power BI**: Microsoft business intelligence

### Cloud Platforms
- **BigQuery**: Google's data warehouse
- **Snowflake**: Cloud data platform
- **Redshift**: Amazon's data warehouse
- **Azure SQL**: Microsoft cloud database

---

## 📚 Resources & References

### Essential Reading
- "Learning SQL" by Alan Beaulieu
- "SQL Cookbook" by Anthony Molinaro
- "High Performance MySQL" by Baron Schwartz
- "PostgreSQL: Up and Running" by Regina Obe

### Online Resources
- W3Schools SQL Tutorial
- SQLBolt interactive lessons
- Mode Analytics SQL Tutorial
- PostgreSQL documentation
- MySQL documentation

### Practice Platforms
- LeetCode SQL problems
- HackerRank SQL challenges
- SQLZoo interactive tutorials
- DataCamp SQL courses
- Kaggle SQL datasets

---

## ✅ Assessment Criteria
- [ ] Query correctness and efficiency
- [ ] Code readability and documentation
- [ ] Problem-solving approach
- [ ] Performance optimization awareness
- [ ] Business logic implementation

---

## 📅 Timeline
**Total Duration**: 16 weeks
**Weekly Commitment**: 6-10 hours
**Prerequisites**: Basic computer literacy
**Next Steps**: Advanced analytics, specific platform specialization