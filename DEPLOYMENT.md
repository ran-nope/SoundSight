# Free deployment

Deploy this repository on [Streamlit Community Cloud](https://share.streamlit.io/).

1. Sign in with GitHub and select **Create app** → **Yup, I have an app**.
2. Select this repository, branch `main`, and entrypoint `app.py`.
3. In Advanced settings, select **Python 3.12**, matching the trained model environment.
4. Deploy. No API key or paid inference service is used.

The repository includes the small fitted `artifacts/model.joblib`, evaluation reports, and a resolved dependency list. `requirements.txt` installs both that environment and the `soundsight` package from `src/`. The full ESC-50 dataset is excluded and is not downloaded or used for training on the hosting server.

Keep the app on the free Community Cloud service. Free hosted apps may sleep when inactive; visitors may need to wait for startup. Training remains a separate local command. Changes pushed to `main` are picked up by Community Cloud.

Reference: [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
