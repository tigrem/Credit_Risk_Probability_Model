# Credit Risk Model

This repository contains the Credit Risk Model for Bati Bank's.

## Credit Scoring Business Understanding

This section summarizes the core concepts of credit risk and their implications for building a credit scoring model at Bati Bank.

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord places significant emphasis on robust risk measurement and management within financial institutions. For a credit scoring model, this translates into a critical need for interpretability and thorough documentation.

* **Interpretability:** Basel II encourages banks to use internal ratings-based (IRB) approaches for calculating capital requirements, which necessitate a deep understanding of how risk parameters (Probability of Default - PD, Loss Given Default - LGD, Exposure at Default - EAD) are derived. An interpretable model allows Bati Bank to clearly understand the factors driving a credit decision, explain these decisions to regulators and customers, and validate the model's logic. It helps in identifying and mitigating biases, ensuring fairness, and demonstrating compliance with regulatory standards. Without interpretability, the model becomes a "black box," making it difficult to justify its outputs or to adapt to changing economic conditions or regulatory requirements.

* **Well-documented Model:** Comprehensive documentation is crucial for regulatory compliance, internal auditing, and knowledge transfer. Basel II requires banks to have sound internal validation processes. Detailed documentation of the model's development, data sources, feature engineering, assumptions, limitations, performance metrics, and ongoing monitoring procedures is essential. This ensures transparency, reproducibility, and accountability, allowing regulators to scrutinize the model's integrity and Bati Bank to maintain a robust risk management framework.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many real-world scenarios, especially with new services or partnerships like Bati Bank's buy-now-pay-later with an eCommerce company, a direct historical "default" label may not be readily available. This necessitates the creation of a **proxy variable** to represent default.

* **Necessity of a Proxy Variable:** A proxy variable is a substitute for the true outcome variable (default) that is difficult or impossible to measure directly. In this case, leveraging behavioral data like Recency, Frequency, and Monetary (RFM) patterns allows us to infer credit risk. For instance, a customer with low recency (haven't bought recently), low frequency (infrequent purchases), and low monetary value (small purchases) might be considered a higher risk, even without a direct default event. This proxy allows us to build a predictive model where a direct label is absent, enabling the launch of the buy-now-pay-later service.

* **Potential Business Risks of Predictions Based on a Proxy:**
    * **Misclassification Risk:** The most significant risk is that the proxy variable may not perfectly align with actual default behavior. A customer classified as "high risk" by the proxy might, in reality, be a "good" customer, leading to missed revenue opportunities. Conversely, a "low risk" classification based on the proxy might lead to extending credit to a customer who eventually defaults, resulting in financial losses.
    * **Suboptimal Model Performance:** If the proxy variable is a poor representation of true default, the model built upon it will have limited predictive power, leading to inaccurate risk assessments and potentially higher default rates than anticipated.
    * **Reputational Damage:** Incorrect credit decisions based on a flawed proxy can lead to customer dissatisfaction, negative publicity, and damage to Bati Bank's and the eCommerce company's reputation.
    * **Regulatory Scrutiny:** If the proxy variable and the resulting model are not well-justified and validated, it could lead to regulatory issues and penalties, especially under frameworks like Basel II that demand robust risk models.
    * **Adverse Selection:** If the model incorrectly identifies low-risk customers as high-risk, it might deter good customers, leaving a higher proportion of genuinely risky customers in the applicant pool.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

In a regulated financial context, the choice between model complexity and interpretability involves significant trade-offs:

* **Simple, Interpretable Models (e.g., Logistic Regression with WoE):**
    * **Pros:**
        * **High Interpretability:** Easy to understand how each feature contributes to the risk score. This is crucial for regulatory compliance (e.g., explaining adverse credit decisions), internal validation, and gaining stakeholder trust. Weight of Evidence (WoE) transformation further enhances interpretability by showing the strength of a predictor's relationship with the target variable.
        * **Transparency and Auditability:** Easier to audit and explain the model's logic to regulators.
        * **Robustness:** Often less prone to overfitting on smaller or noisy datasets.
        * **Computational Efficiency:** Generally faster to train and deploy.
    * **Cons:**
        * **Lower Predictive Performance:** May not capture complex, non-linear relationships in the data as effectively as more complex models, potentially leading to lower accuracy in predicting default.
        * **Limited Feature Interaction Handling:** May struggle to model intricate interactions between features without explicit engineering.

* **Complex, High-Performance Models (e.g., Gradient Boosting):**
    * **Pros:**
        * **Higher Predictive Performance:** Often achieve superior accuracy by capturing complex patterns, non-linear relationships, and feature interactions in the data. This can lead to more precise risk assessments and potentially lower default rates.
        * **Handles Complex Data:** Well-suited for large datasets with many features and intricate relationships.
    * **Cons:**
        * **Lower Interpretability ("Black Box"):** Difficult to understand the exact reasoning behind a prediction, making it challenging to explain to regulators or customers. This can be a major hurdle in a highly regulated environment.
        * **Reduced Transparency and Auditability:** Harder to audit and validate the model's internal workings, which can raise concerns for regulators.
        * **Higher Risk of Overfitting:** More prone to overfitting, especially with insufficient data or improper tuning, leading to poor generalization on unseen data.
        * **Increased Computational Complexity:** Can be more computationally intensive to train and deploy, requiring more resources.
        * **Regulatory Challenges:** Regulators may be hesitant to approve models that lack transparency, potentially requiring significant effort to provide sufficient justification and validation.

**Trade-off Summary:** In a regulated financial context, there's a constant tension between achieving the highest possible predictive accuracy and maintaining the necessary level of interpretability and transparency for regulatory compliance and business understanding. While complex models might offer better performance, the ability to explain and justify every credit decision, as mandated by regulations like Basel II, often pushes financial institutions towards more interpretable models or requires significant effort to build explainability frameworks around complex models. The optimal choice often involves a balance, perhaps starting with interpretable models and gradually incorporating more complex approaches with robust explainability and validation strategies.