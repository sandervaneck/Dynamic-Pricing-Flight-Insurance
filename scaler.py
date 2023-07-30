import openpyxl
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def my_scaler(delay_df, from_col_name, to_col_name, cat, workbook, sheet, row, file):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pd.DataFrame(delay_df[from_col_name],index=delay_df.index))
    scaled_delay_df = pd.DataFrame(scaled, columns=['scale_temp'], index=delay_df.index)
    delay_df[to_col_name] = scaled_delay_df['scale_temp']
    avg_scaled = delay_df.groupby(cat)[to_col_name].mean()
    plt.figure(figsize=(10, 6))
    avg_scaled.plot(kind='bar', color='skyblue')
    plt.xlabel(f"{cat}")
    plt.ylabel(f"Average Scaled {cat} Score")
    plt.title(f"Average Scaled Carrier Score per {cat}")
    temp_file = f'{row}.png'
    plt.savefig(temp_file)
    plt.close()
    img = openpyxl.drawing.image.Image(temp_file)
    plt.close()
    # image_buffer.seek(0)
    # img = openpyxl.drawing.image.Image(image_buffer)
    sheet.add_image(img, f'A{row}')
    workbook.save(file)

