
from fastapi import FastAPI
from spoofApi import app as spoof_app
from registerApi import app as register_app
from authorizeApi import app as authorize_app

#from spoofVoiceApi import app as voice_spoof_app
#from registerVoiceApi import app as voice_register_app
#from authorizeVoiceApi import app as voice_authorize_app
#from convertVoice import app as voice_convert_app

app = FastAPI()

app.mount("/spoof_check", spoof_app)
app.mount("/register", register_app)
app.mount("/authorize", authorize_app)

#app.mount("/predict_voice_spoof", voice_spoof_app)
#app.mount("/register_voice", voice_register_app)
#app.mount("/authorize_voice", voice_authorize_app)
#app.mount("/convert_voice", voice_convert_app)

@app.get("/")
async def root():
    return {"message": "Face + Voice Authentication API Server Running"}
