🩸 𝗠𝗮𝗹𝗮𝗿𝗶𝗮 𝗖𝗲𝗹𝗹 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻 𝘂𝘀𝗶𝗻𝗴 𝗗𝗲𝗲𝗽 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴
An AI-powered computer vision system that classifies microscope images of blood smears as either Parasitized or Uninfected, helping automate early malaria diagnosis.

🧠 𝗢𝘃𝗲𝗿𝘃𝗶𝗲𝘄
Malaria remains a leading cause of illness and death in developing regions.
Traditional diagnosis is manual, slow, and error-prone.

This project uses deep learning to assist medical professionals with faster, more accurate, and consistent malaria detection.

⚙️ 𝗠𝗼𝗱𝗲𝗹𝘀 𝗨𝘀𝗲𝗱
Two transfer learning architectures were implemented:
  • ResNet50
  • VGG19
Both models were fine-tuned on malaria cell images.

📊 𝗘𝘃𝗮𝗹𝘂𝗮𝘁𝗶𝗼𝗻 𝗠𝗲𝘁𝗿𝗶𝗰𝘀
To ensure reliability, multiple metrics were used to assess performance:
 • 𝗣𝗿𝗶𝗺𝗮𝗿𝘆 𝗺𝗲𝘁𝗿𝗶𝗰: Recall (to minimize false negatives)
 • 𝗦𝗮𝘁𝗶𝗳𝗶𝗰𝗶𝗻𝗴 𝗺𝗲𝘁𝗿𝗶𝗰𝘀: F1-score, Accuracy, Precision

🚀 𝗠𝗼𝗱𝗲𝗹 𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲
𝗥𝗲𝘀𝗡𝗲𝘁𝟱𝟬
 • Recall: 𝟵𝟳.𝟴𝟮%
 • Accuracy: 𝟵𝟳.𝟮𝟴%
 • Precision: 𝟵𝟲.𝟳𝟳%
 • F1-score: 𝟵𝟳.𝟮𝟵%

𝗩𝗚𝗚𝟭𝟵
 • Recall: 𝟵𝟳.𝟳𝟭%
 • Accuracy: 𝟵𝟳.𝟮𝟴%
 • Precision: 𝟵𝟲.𝟴𝟳%
 • F1-score: 𝟵𝟳.𝟮𝟵%

 🌐 𝗗𝗲𝗽𝗹𝗼𝘆𝗺𝗲𝗻𝘁
The model was deployed on 𝗦𝘁𝗿𝗲𝗮𝗺𝗹𝗶𝘁, offering an interactive web interface for uploading and classifying cell images.
🔗 𝗧𝗿𝘆 𝗶𝘁 𝗹𝗶𝘃𝗲: https://predictingmalaria1.streamlit.app/
💻 𝗚𝗶𝘁𝗛𝘂𝗯 𝗥𝗲𝗽𝗼𝘀𝗶𝘁𝗼𝗿𝘆: https://github.com/Sadim-Verse/predicting_malaria
