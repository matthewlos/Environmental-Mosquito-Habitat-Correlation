# GeoDude
Explaining how environmental factors correlate with mosquito habitats.

## Features

**Interactive Map Visualization** - Explore global mosquito habitat data.

**Dataset Explorer** - Browse and filter 43,000+ mosquito observations. 

**AI-Powered Insights** - Get intelligent analysis using OpenAI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt

Create google cloud account, and add google Earth Engine API to the New Project
earthengine authenticate
earthengine set_project PROJECT_ID
```

2. Configure your `.env` file with OpenAI API credentials:
```env
OPENAI_API_KEY=your_api_key_here
MODEL=gpt-4o
# Optional: Custom base URL (for institutional proxies)
# OPENAI_BASE_URL=https://api.ai.it.ufl.edu
```

**Note**: If you don't have an OpenAI API key, you can still use the Data Explorer and Interactive Map tabs

## Usage

### Streamlit Web App (Recommended)

Run the interactive web application:
```bash
streamlit run app.py
```

The app includes three main tabs:
- **Data Explorer**: View dataset statistics and preview records
- **Interactive Map**: Visualize mosquito habitats on a global map
- **AI Insights**: Get intelligent analysis with predefined or custom queries

## AI Analysis Types

The tool supports several analysis modes:

1. **General Overview** - Comprehensive dataset insights
2. **Geographic Patterns** - Spatial distribution analysis
3. **Species Distribution** - Species diversity and prevalence
4. **Temporal Trends** - Seasonal and long-term patterns
5. **Custom Query** - Ask specific questions about the data

## Example Queries

- "What are the most common water sources for mosquito breeding?"
- "Which countries have the highest mosquito larvae counts?"
- "How does elevation correlate with mosquito species diversity?"
- "What seasonal patterns exist in mosquito observations?"
- "Which species are most commonly found in artificial water sources?"
