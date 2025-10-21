screen -S llm_fc_hypo_api -d -m bash -c "uvicorn api:app --port 8500"
screen -S llm_fc_hypo_gradio -d -m bash -c "python app_gradio.py"