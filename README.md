# KnowVis: Knowledge-Centric Visual Summarization for Video Lectures ([Paper](https://arxiv.org/pdf/2609.03742), [Dataset](https://huggingface.co/datasets/yixu-cityu/KnowVis))

**KnowVis** is a framework that transforms video lectures into pedagogically grounded visual narratives.

<p align="center">
  <img src="assets/framework_overview.png" width="100%" alt="KnowVis Framework Overview">
</p>

---

## Dataset

[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/yixu-cityu/KnowVis)


We collected **125 video lectures** sourced from OER Commons and YouTube, spanning 10 distinct academic disciplines. From these multimodal inputs, we utilized KnowVis framwork to generate **1,079** pedagogically structured visual summaries. Each visual summary is paired with its source transcripts, slide frames, and concept subgraph. 
You can access and download the dataset directly from this **Hugging Face** [link](https://huggingface.co/datasets/yixu-cityu/KnowVis):



## How to run

Follow these steps to run KnowVis and generate pedagogically grounded visual summaries for key concepts in any video lecture.


### Input Video Format

Use an `.mp4` file path o a `YouTube URL`.

Then edit `config.json`:

Example:
```json
{
    "GEMINI_API_KEY": "YOUR_GEMINI_API_KEY",
    "target_video": {
        "local_path": "",
        "youtube_url": "https://www.youtube.com/watch?v=KS5zllVsz3I"
    }
}
```


### Environment Setup

Use Python **3.11 or 3.12**.

```powershell
pip install -r requirements.txt
```

### Run

```powershell
python run.py --config config.json
```

| Output | Contents |
| --- | --- |
| `knowledge_units.json` | Concept subgraph of each Knowledge Units |
| `knowledge_units/` | Retrieved source content for each Knowledge Units |
| `visuals/` | Generated Visual Summaries |

