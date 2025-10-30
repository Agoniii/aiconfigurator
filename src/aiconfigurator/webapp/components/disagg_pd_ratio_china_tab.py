# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.components.base import create_model_name_config,create_pd_system_config, create_model_quant_config, create_model_parallel_config, create_runtime_config, create_model_misc_config
import gradio as gr
from aiconfigurator.sdk.common import ColumnsDisaggPD

def create_disagg_pd_ratio_china_tab(app_config):
    with gr.Tab("Disaggregation P/D Ratio China"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**
                    This mode is used to analyze P/D ratio in disaggregated system for China configurations.

                    **Features:**
                    - Simplified configuration using preset system combinations (e.g., h20:h20, 6kd:h20)
                    - Uses trtllm backend with default optimized settings
                    - Sweeps multiple TTFT and TPOT values automatically
                    - Provides P/D worker ratio recommendations
                '''
            )

        # create PD system config
        system_choices = ['h20:h20', '6kd:h20', '6kd:6kd']
        with gr.Accordion("PD System"):
            pd_system = gr.Dropdown(choices=system_choices,
                                label="Prefill/Decode System",
                                value='h20:h20',
                                interactive=True)
        model_system_components = {"pd_system": pd_system}
        model_name_components = create_model_name_config(app_config)
        # create runtime config
        with gr.Accordion("Runtime config"):
            with gr.Row():
                isl = gr.Number(value=2048, label='input sequence length', interactive=True)
                osl = gr.Number(value=128, label='output sequence length', interactive=True)
        runtime_config_components = {"isl": isl, "osl": osl}

        estimate_btn = gr.Button("Estimate Prefill/Decode Ratio", variant="primary")

        record_df = gr.Dataframe(
            label="Prefill/Decode Worker Ratio Results:",
            headers=ColumnsDisaggPD,
            interactive=False,
            wrap=True
        )

        pivot_df = gr.Dataframe(
            label="P/D Ratio Pivot Table (Rows: TTFT, Columns: TPOT, Values: pd_ratio):",
            interactive=False,
            wrap=True
        )

        debugging_box = gr.Textbox(label="Debugging Output", lines=10, max_lines=20)

        with gr.Row():
            download_btn = gr.Button("Download")
        output_file=gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        'model_name_components': model_name_components,
        'runtime_config_components': runtime_config_components,
        'model_system_components': model_system_components,
        'estimate_btn': estimate_btn,
        'record_df': record_df,
        'pivot_df': pivot_df,
        'debugging_box': debugging_box,
        'download_btn': download_btn,
        'output_file': output_file,
        'introduction': introduction
    }