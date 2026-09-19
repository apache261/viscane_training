**VISCANE: An Edge-Based AI for Sugarcane Quality Grading and Maturity Detection Using RGB-HSV Imaging** 

**Faye N. Cuevas\*, Windy D. Elipse, John Rolo C. Flores,**   
**Lynol I. Ibarra**  
College of Information and Communications Technology, STI West Negros University, Bacolod City, Philippines  
Email: [cuevas.247492@wnu.sti.edu.ph](mailto:cuevas.247492@wnu.sti.edu.ph)

**Abstract (12-point bold font on single line spacing)**  
For abstract content, use 12-point Times New Roman font on single line spacing. First line is indented 0.5 inch.  An abstract of up to **250 words** must be included. Include your major findings in a useful and concise manner. Include a problem statement, objectives, brief methods, results, and the significance of your findings.

***Keywords*****:** *List up to 5 keywords and separate each keyword by a semi-colon (;).  The keywords should accurately reflect the content of the article.  The keywords will be used for indexing purposes.*

# 

# **1\.  Introduction**

Sugarcane (Saccharum officinarum) is one of the world’s most important crops. It produces over 1.9 billion metric tons annually. It is a major source of sugar, bioethanol, and biomass energy (Food and Agriculture Organization of the United Nations, 2025). According to Rose and Chilvers (2018), farmers face challenges such as climate change, resource scarcity, and market instability. This led to the adoption of digital tools, such as artificial intelligence and precision farming, to improve productivity and sustainability.   
In Southeast Asia, sugarcane supports rural employment and regional growth, but it still faces pressures from changing weather, labor shortages, and weak supply chains (Solomon et al., 2024). Countries such as Thailand, the Philippines, and Indonesia are adopting technologies, namely satellite imaging, mobile advisory services, and blockchain systems, to improve transparency (Diyanah, Kozono, and Hazmi, 2024). Despite these advances, rural adoption remains low due to poor infrastructure and limited digital skills (Montesclaros and Teng, 2023).  
In the Philippines, sugarcane covers more than 400,000 hectares, with 85% of farmers cultivating plots of five hectares or less (Sugar Regulatory Administration \[SRA\], 2019). This smallholder dominance leaves the industry vulnerable to unstable yields and market price volatility. Programs like block farming and digital agriculture have shown potential to increase productivity and incomes (Pantoja et al., 2019; Briones et al., 2023). However, adoption remains limited due to the need for inclusive policies and cooperative systems (Oñal, Junitilla, and Ortega, 2021).  
Negros Occidental, known as the “Sugar Bowl of the Philippines,” produces more than half of the country’s sugar. Yet smallholder farmers continue to struggle with declining yields, rising costs, and climate risks (Briones, Galang, and Latigar, 2023). Reliance on traditional practices and limited access to digital tools reduce competitiveness, which makes technology adoption essential to sustain livelihoods and protect the province’s role in the national sugar industry (SRA, 2019).   
To address these challenges, this study introduces VISCANE: An Edge-Based AI System for Sugarcane Quality Grading and Maturity Detection Using RGB-HSV Imaging.   
VISCANE integrates farmer-reported inputs with image-based maturity classification to generate real-time estimated predictions of sugar content. Specifically, the system employs YOLOv8 for real-time detection of sugarcane stalks and ResNet for deep feature extraction and maturity-level classification. By combining these models, VISCANE achieves both speed and accuracy in computer vision analysis and ensures reliable grading outcomes.  
By means of offering a low-cost, non-destructive, and accessible tool, the system empowers smallholder farmers, guides them in assessing sugarcane quality before and after harvest, and strengthens their participation in the sugarcane value chain. This research contributes to inclusive digital transformation and aligns with national and ASEAN priorities for resilience and competitiveness.

**2\.  Objectives**  
The main objective of this study is to develop VISCANE, an edge-based AI system for sugarcane quality grading and maturity assessment using RGB-HSV imaging. The system is designed to provide smallholder farmers with a non-destructive, accessible decision-support tool that improves grading accuracy and guides farmers in evaluating the quality of sugarcane before and after harvesting.  
Specifically, the study aims to:

* Design and implement an edge-based AI architecture capable of processing RGB-HSV imaging data for sugarcane quality and maturity grading.  
* Establish a non-destructive imaging methodology that guarantees transparency, operational reliability, and ease of use for smallholder farmers operating in rural environments.  
*  Integrate beneficiary-centered features into the system to support equitable valuation and reliable crop assessment, specifically through:  
  1. Inclusion of essential agronomic inputs, such as crop variety, total land area, and infestation severity;  
  2. Application of computer vision for maturity detection to facilitate objective crop grading (e.g., classifying stalks as “Mature”);  
  3. Utilization of model-based yield calculations anchored on baseline LKG/TC and LKG/HA data from the Sugar Regulatory Administration (SRA), mathematically adjusted for agronomic penalties (e.g., a localized penalty for RSSI infestation);  
  4. Computation of area-weighted LKG/TC and LKG/HA averages to project realistic, farm-level production outcomes;  
  5.  Generation of predictive estimated yield outputs that translate the user's input land area (in hectares) into actionable Total LKG forecasts (e.g., 25 tons × 1.87 LKG/TC \= 46.75 LKG); and  
  6.  Development of transparent reporting interfaces that display these calculations in a clear, accessible format to guide farmer evaluation during pre- and post-harvest stages.  
* Evaluate the system’s effectiveness and efficiency in both pre-harvest and post-harvest settings, ensuring alignment with local realities, improved decision-making, and farmer empowerment through accessible and actionable insights.
