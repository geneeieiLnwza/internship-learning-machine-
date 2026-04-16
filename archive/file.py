import kagglehub
import pandas as pd
import os
import glob

path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
print("Path to dataset files:", path)

csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)

if not csv_files:
    print("❌ ไม่เจอไฟล์ CSV ในโฟลเดอร์ที่โหลดมาเลยครับ ลองเช็คดูใหม่นะ")
else:
    target_file = csv_files[0]
    print(f"✅ เจอไฟล์แล้ว: {target_file}")
    df = pd.read_csv(target_file)
    
    print("\n--- ข้อมูล 5 แถวแรก ---")
    print(df.head())