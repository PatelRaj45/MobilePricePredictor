import gradio as gr
import joblib
import pandas as pd
from model import predict_price_range

# Load class labels
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Prediction function
def gradio_predict_ui(
    battery_power, blue, clock_speed, dual_sim, fc, four_g,
    int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
    px_width, ram, sc_h, sc_w, talk_time, three_g,
    touch_screen, wifi
):
    input_data = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g,
        int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
        px_width, ram, sc_h, sc_w, talk_time, three_g,
        touch_screen, wifi
    ]], columns=[
        'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
        'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
        'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
        'touch_screen', 'wifi'
    ])

    pred_class, probs = predict_price_range(input_data)
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("📱 **Mobile Price Range Predictor**")
    gr.Markdown("Predict the price category of a smartphone based on specifications (0 = Low, 3 = Very High).")

    with gr.Row():
        with gr.Column():
            battery_power = gr.Slider(500, 2000, step=10, label="Battery Power (mAh)")
            blue = gr.Radio([0, 1], label="Bluetooth")
            clock_speed = gr.Slider(0.5, 3.0, step=0.1, label="Clock Speed (GHz)")
            dual_sim = gr.Radio([0, 1], label="Dual SIM")
            fc = gr.Slider(0, 20, step=1, label="Front Camera (MP)")
            four_g = gr.Radio([0, 1], label="4G Support")
            int_memory = gr.Slider(4, 128, step=4, label="Internal Memory (GB)")
            m_dep = gr.Slider(0.1, 1.0, step=0.1, label="Mobile Depth (cm)")
            mobile_wt = gr.Slider(80, 250, step=5, label="Mobile Weight (g)")
            n_cores = gr.Slider(1, 8, step=1, label="No. of CPU Cores")
            pc = gr.Slider(2, 20, step=1, label="Primary Camera (MP)")

        with gr.Column():
            px_height = gr.Slider(100, 1960, step=10, label="Pixel Height")
            px_width = gr.Slider(500, 2000, step=10, label="Pixel Width")
            ram = gr.Slider(256, 8192, step=128, label="RAM (MB)")
            sc_h = gr.Slider(5, 20, step=1, label="Screen Height (cm)")
            sc_w = gr.Slider(2, 15, step=1, label="Screen Width (cm)")
            talk_time = gr.Slider(2, 20, step=1, label="Talk Time (hrs)")
            three_g = gr.Radio([0, 1], label="3G Support")
            touch_screen = gr.Radio([0, 1], label="Touch Screen")
            wifi = gr.Radio([0, 1], label="WiFi")

    submit_btn = gr.Button("🔍 Predict Price Range")
    output_label = gr.Label(num_top_classes=4, label="Predicted Price Range")

    submit_btn.click(
        fn=gradio_predict_ui,
        inputs=[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                px_width, ram, sc_h, sc_w, talk_time, three_g,
                touch_screen, wifi],
        outputs=output_label
    )

    # 4 Examples - one from each class (0-3)
    gr.Examples(
        examples=[
            [800, 1, 1.2, 1, 2, 0, 8, 0.4, 100, 2, 5, 300, 500, 512, 10, 5, 8, 0, 0, 0],        # Low Cost
            [1400, 1, 1.8, 1, 8, 1, 32, 0.6, 120, 4, 13, 800, 900, 2048, 13, 6, 12, 1, 1, 1],   # Medium Cost
            [1453, 0, 1.6, 1, 12, 1, 52, 0.3, 96, 2, 18, 187, 1311, 2373, 10, 1, 10, 1, 1, 1],  # High Cost
            [1900, 1, 2.9, 1, 16, 1, 128, 1.0, 180, 8, 20, 1800, 1920, 6144, 16, 10, 20, 1, 1, 1] # Very High Cost
        ],
        inputs=[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                px_width, ram, sc_h, sc_w, talk_time, three_g,
                touch_screen, wifi],
        outputs=output_label,
        fn=gradio_predict_ui,
        run_on_click=True
    )

# Launch app
demo.launch()
