import openpyxl
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns


def my_scaler(delay_df, from_col_name, to_col_name, cat, workbook, sheet, row, file):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pd.DataFrame(delay_df[from_col_name],index=delay_df.index))
    scaled_delay_df = pd.DataFrame(scaled, columns=['scale_temp'], index=delay_df.index)
    delay_df[to_col_name] = scaled_delay_df['scale_temp']
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=delay_df, x=cat, y=to_col_name, color='skyblue')
    plt.xlabel(f"{cat}")
    plt.ylabel(f"Scaled {cat} Score")
    plt.title(f"Distribution of Scaled Carrier Scores per {cat}")
    temp_file = f'{row}.png'
    plt.savefig(temp_file)
    plt.close()
    img = openpyxl.drawing.image.Image(temp_file)
    sheet.add_image(img, f'A{row}')
    workbook.save(file)


