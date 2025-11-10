# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr

from aiconfigurator.sdk.common import ColumnsDisaggPD
from aiconfigurator.webapp.components.base import (
    create_model_name_config,
)


def create_disagg_pd_ratio_h20_6kd_tab(app_config):
    with gr.Tab("Disaggregation P/D Ratio for H20 and 6KD"):
        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r"""
                    **Please read the readme tab before using this tab.**
                    This mode is used to analyze P/D ratio in disaggregated system for H20 and 6KD configurations.

                    **Features:**
                    - Simplified configuration using preset system combinations (e.g., h20:h20, 6kd:h20)
                    - Uses trtllm backend with default settings:
                        - gemm_quant_mode: fp8
                        - kvcache_quant_mode: fp8
                        - fmha_quant_mode: float16
                        - moe_quant_mode: fp8
                        - comm_quant_mode: fp8
                        - nextn: 1 for deepseek, other models are 0
                        - nextn_accept_rates: [0.85,0,0,0,0]
                    - Supports custom TTFT and TPOT values (comma-separated for multiple values, e.g., 300,600,1000)
                    - Provides P/D worker ratio for the best configuration that meets the TTFT and TPOT constraints
                        - ttft <= ttft_limit
                        - tpot in range of [tpot_limit * 0.9, tpot_limit * 1.2]
                """,
            )

        # create PD system config
        system_choices = ["h20:h20", "h20:6kd", "6kd:h20", "6kd:6kd"]
        with gr.Accordion("PD System"):
            pd_system = gr.Dropdown(
                choices=system_choices, label="Prefill/Decode System", value="h20:h20", interactive=True
            )
        model_system_components = {"pd_system": pd_system}
        model_name_components = create_model_name_config(app_config)
        # create runtime config
        with gr.Accordion("Runtime config"):
            with gr.Row():
                isl = gr.Number(value=2048, label="input sequence length", interactive=True)
                osl = gr.Number(value=128, label="output sequence length", interactive=True)
        with gr.Accordion("Limitations config"):
            with gr.Row():
                ttft = gr.Textbox(
                    value="300,1000,1500",
                    label="TTFT limitations (ms, comma-separated)",
                    interactive=True,
                    placeholder="e.g., 300,1000,1500"
                )
                tpot = gr.Textbox(
                    value="10,20,30,50",
                    label="tokens/s/user limitations (comma-separated)",
                    interactive=True,
                    placeholder="e.g., 10,20,30,50"
                )
        runtime_config_components = {"isl": isl, "osl": osl, "ttft": ttft, "tpot": tpot}

        estimate_btn = gr.Button("Estimate Prefill/Decode Ratio", variant="primary")

        results_df = gr.Dataframe(
            label="Results DataFrame", headers=["index"] + ColumnsDisaggPD, interactive=False, wrap=True
        )
        pivot_df = gr.Dataframe(
            label="P/D Ratio Pivot Table (Rows: TTFT, Columns: tokens/s/user, Values: pd_ratio):",
            interactive=False,
            wrap=True,
            elem_classes="equal-width-columns"
        )

        debugging_box = gr.Textbox(label="Debugging Output", lines=10, max_lines=20)

        with gr.Row():
            download_btn = gr.Button("Download")
        output_file = gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        "model_name_components": model_name_components,
        "runtime_config_components": runtime_config_components,
        "model_system_components": model_system_components,
        "estimate_btn": estimate_btn,
        #"results_df": results_df,
        "pivot_df": pivot_df,
        "debugging_box": debugging_box,
        "download_btn": download_btn,
        "output_file": output_file,
        "introduction": introduction,
    }
