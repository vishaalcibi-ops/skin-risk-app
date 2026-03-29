"""Quick script to generate PWA icons from source image."""
import os, shutil

src = r'C:\Users\ADMIN\.gemini\antigravity\brain\f586ab7b-bded-47d4-a27d-949d51685019\skin_risk_icon_1772780964128.png'
out_dir = os.path.join(os.path.dirname(__file__), 'static', 'icons')
os.makedirs(out_dir, exist_ok=True)

# Copy original as both sizes (will be rendered at correct size by browsers via manifest)
shutil.copy2(src, os.path.join(out_dir, 'icon-512.png'))
shutil.copy2(src, os.path.join(out_dir, 'icon-192.png'))
print(f'Icons copied to {out_dir}')
print(os.listdir(out_dir))
