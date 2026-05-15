# Personal Recording Guide — Tamil ASR Test Samples

Record each sentence below and save the file as indicated.
Speak clearly at a natural pace. No need to be slow or robotic.

**Recording tips:**
- Use your phone voice recorder or any mic
- Quiet room, no background music
- Save as WAV or MP3
- File names match what's listed — drop them in this folder (`sample_inputs/personal/`)

---

## Set 1 — Everyday Sentences (Short)

| # | File name | Tamil sentence | Meaning |
|---|-----------|----------------|---------|
| 1 | `p01_greeting.wav` | வணக்கம், எப்படி இருக்கீங்க? | Hello, how are you? |
| 2 | `p02_name.wav` | என் பெயர் தினேஷ். | My name is Dinesh. |
| 3 | `p03_today.wav` | இன்று வெயில் அதிகமாக இருக்கு. | It is very sunny today. |
| 4 | `p04_food.wav` | சாப்பிட்டீங்களா? | Have you eaten? |
| 5 | `p05_time.wav` | இப்போ என்ன நேரம்? | What is the time now? |

---

## Set 2 — Medium Sentences

| # | File name | Tamil sentence | Meaning |
|---|-----------|----------------|---------|
| 6 | `p06_work.wav` | நான் கணினி அறிவியல் படிக்கிறேன். | I am studying computer science. |
| 7 | `p07_travel.wav` | சென்னைக்கு பஸ்ல போவது நல்லது. | It is good to go to Chennai by bus. |
| 8 | `p08_weather.wav` | தமிழ்நாட்டில் மழை காலம் அக்டோபரில் வருகிறது. | The rainy season in Tamil Nadu comes in October. |
| 9 | `p09_news.wav` | இன்றைய செய்திகளில் என்ன நடந்தது? | What happened in today's news? |
| 10 | `p10_study.wav` | தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்று. | Tamil is one of the oldest languages in the world. |

---

## Set 3 — Numbers and Dates

| # | File name | Tamil sentence | Meaning |
|---|-----------|----------------|---------|
| 11 | `p11_numbers.wav` | ஒன்று, இரண்டு, மூன்று, நான்கு, ஐந்து. | One, two, three, four, five. |
| 12 | `p12_price.wav` | இந்த புத்தகத்தின் விலை நூறு ரூபாய். | The price of this book is one hundred rupees. |
| 13 | `p13_date.wav` | இன்று மே பதினைந்தாம் தேதி. | Today is the fifteenth of May. |

---

## Set 4 — Longer / Challenging (Tests robustness)

| # | File name | Tamil sentence | Meaning |
|---|-----------|----------------|---------|
| 14 | `p14_long.wav` | இந்தியாவில் பல மொழிகள் பேசப்படுகின்றன, ஆனால் தமிழ் மொழி தனிச்சிறப்பு வாய்ந்தது. | Many languages are spoken in India, but Tamil holds a unique distinction. |
| 15 | `p15_code_mix.wav` | நான் Python programming கத்துக்கிட்டிருக்கேன். | I am learning Python programming. |
| 16 | `p16_noise_test.wav` | தமிழகத்தில் பல்வேறு கலாச்சாரங்கள் கலந்து வாழ்கின்றன. | Many cultures live together in Tamil Nadu. |

---

## After Recording

1. Save all files into `sample_inputs/personal/`
2. Run the app: `python -m app.main`
3. Upload each file, try both **local Whisper** and **Groq Cloud** models
4. Note which model gets the sentence right
5. Add your observations to `evaluation/personal_results.md` (see evaluation folder)

---

## Ground Truth Reference

Use the sentences in the table above as ground truth when comparing model outputs.
For a rough WER check, compare the model output word-by-word against the Tamil sentence.
