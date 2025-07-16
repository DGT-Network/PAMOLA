### **TRANSACTIONS DATASET (BANKING)**

**Base URL**: [https://github.com/ahsan084/Banking-Dataset/blob/main/Comprehensive\_Banking\_Database.csv](https://github.com/ahsan084/Banking-Dataset/blob/main/Comprehensive_Banking_Database.csv)
**Current Name**: DATA/raw/bank_txs/txs.csv
**License**: MIT License

**Description**:
This dataset provides a comprehensive view of banking customer behavior, combining personal, transactional, loan, credit card, and feedback information. It is structured as a tabular dataset suitable for testing privacy-preserving data generation, anonymization algorithms, and various machine learning models in the financial domain. Each row represents a customer's profile enriched with multivariate information collected over time.

---

### **Dataset Features**:

**Personal and Demographic Information**

* `Customer ID`: Unique identifier for each customer.
* `First Name`, `Last Name`: Identifiable name fields (PII).
* `Age`: Customer's age.
* `Gender`: Gender of the customer.
* `Address`, `City`: Residential information.
* `Contact Number`, `Email`: Direct identifiers.

**Account Information**

* `Account Type`: Type of bank account (e.g., Savings, Current).
* `Account Balance`: Balance prior to the last transaction.
* `Date Of Account Opening`: When the account was initiated.
* `Last Transaction Date`: Timestamp of the most recent transaction.

**Transactional Data**

* `TransactionID`: Unique transaction reference.
* `Transaction Date`: Date of the transaction.
* `Transaction Type`: Nature of the transaction (e.g., Deposit, Withdrawal, Transfer).
* `Transaction Amount`: Amount involved.
* `Account Balance After Transaction`: Post-transaction balance.
* `Branch ID`: Identifies the branch where the transaction occurred.

**Loan Details**

* `Loan ID`: Identifier of any active or historic loan.
* `Loan Amount`: Principal amount.
* `Loan Type`: Loan classification (Mortgage, Auto, etc.).
* `Interest Rate`: Associated rate for the loan.
* `Loan Term`: Duration of the loan.
* `Approval/Rejection Date`: Decision date.
* `Loan Status`: Final status (Approved, Rejected, Closed).

**Credit Card Information**

* `Card ID`: Unique card identifier.
* `Card Type`: e.g., AMEX, MasterCard, Visa.
* `Credit Limit`: Assigned credit ceiling.
* `Credit Card Balance`: Outstanding balance.
* `Minimum Payment Due`: Due amount.
* `Payment Due Date`: Next payment deadline.
* `Last Credit Card Payment Date`: Date of last payment.
* `Rewards Points`: Points accumulated.

**Feedback & Anomaly Monitoring**

* `Feedback ID`, `Feedback Date`, `Feedback Type`: Captures customer support and complaints.
* `Resolution Status`, `Resolution Date`: Tracks issue resolution.
* `Anomaly`: Binary indicator (e.g., 1 for anomaly detected).

---

### **Usage**:

This dataset supports the following applications:

* **Anonymization & Risk Modeling**: Ideal for testing k-anonymity, l-diversity, differential privacy, and synthetic data pipelines.
* **Fraud Detection**: Enables pattern detection and anomaly modeling in transactional sequences.
* **Customer Churn & Segmentation**: Can support clustering and classification tasks.
* **Synthetic Data Benchmarking**: Acts as a rich tabular base for evaluating privacy-preserving data generators such as PATE-GAN and DP-GAN.

---

### **Anonymization Considerations**:

This dataset contains **direct identifiers (PII)** (e.g., names, emails, contact numbers), **quasi-identifiers** (e.g., Age, City, Gender), and **sensitive attributes** (e.g., Loan Status, Transaction Amount). As such, it is highly suitable for testing the effectiveness of anonymization and privacy metrics under realistic enterprise conditions—especially within banking use cases.

