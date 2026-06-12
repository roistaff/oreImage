import asyncio
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from PIL import Image

app = FastAPI(title="Creative LCM Realtime API")
model_lock = asyncio.Lock()

# CORS設定（どこからでもアクセス可能に）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Stable LCM Model...")
# 発想力豊かで、色彩感覚が素晴らしい汎用アートモデルを使用
pipe = AutoPipelineForImage2Image.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# スケジューラーをLCM用に換装
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
print("System Ready!")

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/generate")
async def generate_image(
    image_file: UploadFile = File(...),
    prompt: str = Form(""),
    strength: float = Form(0.70)
):
    request_object_content = await image_file.read()
    input_image = Image.open(io.BytesIO(request_object_content)).convert("RGB").resize((512, 512))

    # 💡 【発想力を豊かにするギミック】
    # ユーザーが入力したプロンプトを活かしつつ、空白や簡易入力でも
    # AIが自発的に美しいアートを連想して補完するベース呪文を結合します。
    base_style = "masterpiece, highly detailed, vivid colors, expressive artwork, fantasy oil painting style, hyperrealistic concept art"
    if prompt.strip():
        final_prompt = f"{prompt.strip()}, {base_style}"
    else:
        final_prompt = base_style

    async with model_lock:
        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt,
                image=input_image,
                num_inference_steps=5,     # ✨ 推論ステップ数を 5 に変更
                guidance_scale=8.5,        # 自由度と呪文のバランスを取る少し高めの設定
                strength=strength,         # フロント側のスライダー値（初期値 0.70）
            ).images[0]

    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
