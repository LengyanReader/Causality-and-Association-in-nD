# Panel App — Deploy to Hugging Face Spaces (Free)

1. Go to https://huggingface.co/new-space
2. Choose:
   - Owner: your username
   - Space name: `causality-ucl-demo`
   - SDK: **Docker**
   - Docker template: **Blank**
3. Upload these files:
   - `panel_app/app.py`
   - `panel_app/requirements.txt`
   - `Dockerfile` (see below)
   - The entire `nomnom/` and `ucl/` directories from the repo root
4. The Space URL will be: `https://huggingface.co/spaces/<your-username>/causality-ucl-demo`

## Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install panel numpy pandas scipy scikit-learn networkx matplotlib plotly
COPY nomnom/ ./nomnom/
COPY ucl/ ./ucl/
COPY panel_app/*.py ./
EXPOSE 7860
CMD ["panel", "serve", "app.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin", "*"]
```

Then embed in your GitHub Pages landing page with an iframe pointing to the HF Space URL.
