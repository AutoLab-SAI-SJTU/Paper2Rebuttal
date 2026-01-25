# RebuttalAgent: AI-Powered Academic Paper Rebuttal Assistant

<p align="center">
  <a href="https://arxiv.org/abs/2601.14171" target="_blank"><img src="https://img.shields.io/badge/arXiv-2601.14171-red"></a>
  <a href='https://mqleet.github.io/Paper2Rebuttal_ProjectPage/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
  <a href="https://huggingface.co/spaces/Mqleet/RebuttalAgent" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red%22"></a>
  <a href="https://huggingface.co/papers/2601.14171" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Daily Papers-red"></a>


</p >


<p align="center">
<strong><big>If you find our work useful, please consider giving us a star 🌟</big></strong>
</p>


## 🔥 News
- [2026.01.21] We release the [paper](https://arxiv.org/abs/2601.14171), code and [project pages](https://mqleet.github.io/Paper2Rebuttal_ProjectPage/)
- [2026.01.23] We create the [WeChat Group](assets/wechat_group.png), welcome to join in and discuss the issues.


## :memo: TODO

- [x] Code Release
- [x] Paper Release
- [ ] Datasets
- [x] Huggingface Demo


## 🔎 Overview

**RebuttalAgent** is an AI-powered multi-agent system that helps researchers craft high-quality rebuttals for academic paper reviews. The system analyzes reviewer comments, searches relevant literature, generates rebuttal strategies, and produces formal rebuttal letters, all through an interactive human-in-the-loop workflow.


### Key Features

- 📄 **Automatic Paper Parsing**: Converts PDF papers to structured text using Docling
- 🔍 **Issue Extraction**: Breaks down reviewer comments into actionable issues with priority levels
- 📚 **Literature Search**: Automatically searches arXiv for relevant supporting papers
- 💡 **Strategy Generation**: Creates data-driven rebuttal strategies (not sophistry!)
- ✍️ **Rebuttal Writing**: Generates formal, conference-ready rebuttal letters
- 🔄 **Human Feedback Loop**: Iteratively refine strategies based on author input



**We recommend to deploy our work locally, following the instructions below. You can also use [Hugginface Space](https://huggingface.co/spaces/Mqleet/RebuttalAgent) to have a try, but it may be very slow，taking up to an hour to finish**



## 🛠️ Installation

### Setup

**Environment**
```bash
conda create -n rebuttal python=3.10
conda activate rebuttal
# if you want to run on GPU , please firstly install the GPU version of PyTorch (select the one based on your CUDA version)
pip install -r requirements.txt
```



## 🚀 Quick Start


### ⚙️ Configuration

**API Key**

Copy the example environment file and fill in your API key for your chosen provider:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:

```bash
# OpenRouter (supports many models)
OPENROUTER_API_KEY=<your_openrouter_api_key>

# Qwen (Alibaba DashScope)
QWEN_API_KEY=<your_qwen_api_key>

# DeepSeek
DEEPSEEK_API_KEY=<your_deepseek_api_key>

# OpenAI
OPENAI_API_KEY=<your_openai_api_key>

# Gemini (Google)
GEMINI_API_KEY=<your_gemini_api_key>

# ZhiPu (GLM)
ZHIPUAI_API_KEY=<your_zhipuai_api_key>
```

**You can alse edit the base_url in .env**

> **Note:** Only configure the provider you plan to use.

### Start with Gradio🤗

> **Note (Windows):** On Windows, you may need to run the terminal as Administrator for the first run, as the PDF parser (Docling) needs to download AI models that require symlink permissions. Alternatively, you can enable [Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development).

> **Performance Note:** Literature analysis involves parsing multiple external papers using **Docling**, which is computationally intensive, we **strongly recommend using GPU acceleration**. 
> - **CPU Mode:** Processing a full workflow may take up to **1 hour**.
> - **GPU Mode (CUDA):** Processing typically takes only about **15 minutes**.


Launch the interactive web interface:

```bash
python app.py --port 8080
```

**With GPU acceleration (for PDF parsing):**
```bash
python app.py --device cuda --port 8080
```

> 🤯 **Note:** The first run may be slower as Docling needs to download AI models from Hugging Face. Subsequent runs will be much faster.

### 🔄 Session Recovery
If your browser page is accidentally refreshed during processing, you can resume your session:
1. Enter your API Key in the configuration section
2. Click **🔄 Refresh List** to see active sessions
3. Select your session from the dropdown
4. Click **▶️ Resume Selected Session**
> ⚠️ Note: If the server restarts, the app will attempt to restore sessions from `gradio_uploads`. This is best-effort and depends on saved logs/summaries, so some in-memory progress may be missing.


### Workflow

1. **Upload Files**: Upload your paper PDF and review text file (raw document including all reviewers' comments and their IDs)
2. **Initial Analysis**: The system parses your paper and extracts reviewer issues
3. **Strategy Review**: For each issue, review the AI-generated rebuttal strategy
4. **Provide Feedback**: Refine strategies through natural language feedback
5. **Generate Rebuttal**: Download the final rebuttal letter and strategy summary

## 📋 Agent Pipeline

The system uses a multi-agent pipeline:

| Agent | Role |
|-------|------|
| **Semantic Encoder** | Compresses paper into high-density format |
| **Issue Extractor** | Extracts and merges reviewer concerns |
| **Literature Retrieval** | Decides if external papers are needed |
| **Reference Filter** | Filters relevant references |
| **Reference Analyzer** | Analyzes supporting literature |
| **Strategy Generator** | Creates rebuttal strategy and to-do list |
| **Strategy Reviewer** | Reviews and improves strategy |
| **Rebuttal Writer** | Writes formal rebuttal letter |
| **Rebuttal Reviewer** | Polishes final rebuttal |


## ✒️ Citation

If you find RebuttalAgent useful for your research, please consider citing:

```bibtex
@misc{ma2026paper2rebuttal,
      title={Paper2Rebuttal: A Multi-Agent Framework for Transparent Author Response Assistance}, 
      author={Qianli Ma and Chang Guo and Zhiheng Tian and Siyu Wang and Jipeng Xiao and Yuanhao Yue and Zhipeng Zhang},
      year={2026},
      eprint={2601.14171},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.14171}, 
}
```

## ❤️ Acknowledgement

### ✍️ RebuttalAgent Contributers

>[Qianli Ma](https://mqleet.github.io/), [Chang Guo](https://github.com/MrC-crm), [Zhiheng Tian](https://github.com/ElysiaTT)

### 🤗 Open-Sourced Repositories

This project builds upon the following open-source projects:
- [Docling](https://github.com/docling-project/docling) - PDF parsing

Thanks to the contributors for their great work!


## 📄 License
This project is licensed under the MIT License.

## 🤝 Community
🥳 Welcome to join our wechat group if you have issues, or want to discuss new ideas!
![WeChat Group](assets/wechat_group.png)
