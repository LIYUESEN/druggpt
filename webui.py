import gradio as gr
import argparse
import psutil
import os
import signal
import platform

p_generate = None
system = platform.system()


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


def kill_process(pid):
    if (system == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def change_input_format(input_format):
    if input_format == "Protein Sequence":
        return {
            "__type__": "update",
            "placeholder": "Input protein amino acid sequence here"
        }
    else:
        return {
            "__type__": "update",
            "placeholder": "Input FASTA file path here"
        }


def disable_atoms_limit(ligand_prompt):
    if ligand_prompt:
        return {"__type__": "update", "value": None, "interactive": False}, {"__type__": "update", "value": None, "interactive": False}
    else:
        return {"__type__": "update", "interactive": True}, {"__type__": "update", "interactive": True}


def run_generate(input_format,
                 input_data,
                 ligand_prompt,
                 number,
                 batch_size,
                 device,
                 output,
                 top_k,
                 top_p,
                 min_atoms,
                 max_atoms
                 ):
    global p_generate
    if p_generate:
        yield "Generation in progress. Please stop the current task before running the next one.", {
            "__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}
    else:
        cmd = f'python drug_generator.py {"-p " if input_format == "Protein Sequence" else "-f"} {input_data} {"-l "+ligand_prompt if ligand_prompt else ""} -n {number} -d {device} -o {output} -b {batch_size} --top_k {top_k} --top_p {top_p} {"--min_atoms "+str(min_atoms) if min_atoms else ""} {"--max_atoms "+str(max_atoms) if max_atoms else ""}'
        yield "Generation started：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        print(cmd)
        p_generate = psutil.Popen(cmd, shell=True)
        p_generate.wait()
        p_generate = None
        yield "Generation completed!", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


def stop_generate():
    global p_generate
    if p_generate:
        kill_process(p_generate.pid)
        p_generate = None
    return "Generation stopped!", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=11451, help='Port of WebUI, default is 11451')

    args = parser.parse_args()

    with gr.Blocks(title="DrugGPT Inference WebUI") as app:
        gr.Markdown(value="# DrugGPT Inference WebUI\nLicensed under GNU GPLv3")
        with gr.Group():
            gr.Markdown(value="Input")
            with gr.Row():
                dropdownInputFormat = gr.Dropdown(label="Input Format",
                                                  choices=["Protein Sequence", "FASTA File"],
                                                  value="Protein Sequence",
                                                  interactive=True)
                textboxInput = gr.Textbox(label="Input Data", interactive=True,
                                          placeholder="Input protein amino acid sequence here")
                textboxLigandPrompt = gr.Textbox(label="Ligand Prompt", interactive=True)

            gr.Markdown(value="Parameters")
            with gr.Row():
                numberAmount = gr.Number(label="Minimum Generation Amount", interactive=True, value=100)
                numberBatchSize = gr.Number(label="Batch Size", interactive=True, value=32)
                sliderTopK = gr.Slider(minimum=1, maximum=100, step=1, label="top_k", value=9, interactive=True)
                sliderTopP = gr.Slider(minimum=0, maximum=1, step=0.05, label="top_p", value=0.9, interactive=True)
                numberMinAtoms = gr.Number(label="Minimum Atoms", interactive=True, value=lambda: None)
                numberMaxAtoms = gr.Number(label="Maximum Atoms", interactive=True, value=lambda: None)
            gr.Markdown(value="Output")
            with gr.Row():
                textboxOutput = gr.Textbox(label="Output Folder", interactive=True, value="ligand_output/")
                dropdownDevice = gr.Dropdown(label="Device",
                                             choices=["cuda", "cpu"],
                                             value="cuda",
                                             interactive=True)
                btnRunGenerating = gr.Button("Run Generating", variant="primary")
                btnStopGenerating = gr.Button("Stop Generating", variant="primary", visible=False)

            gr.Markdown(value="Log")
            with gr.Row():
                textboxLog = gr.Textbox(label="Running Status")

            btnRunGenerating.click(run_generate,
                                   [dropdownInputFormat, textboxInput, textboxLigandPrompt, numberAmount,
                                    numberBatchSize, dropdownDevice, textboxOutput, sliderTopK, sliderTopP, numberMinAtoms, numberMaxAtoms],
                                   [textboxLog, btnRunGenerating, btnStopGenerating])
            btnStopGenerating.click(stop_generate, [], [textboxLog, btnRunGenerating, btnStopGenerating])

            dropdownInputFormat.change(change_input_format, [dropdownInputFormat], [textboxInput])
            textboxLigandPrompt.change(disable_atoms_limit, [textboxLigandPrompt], [numberMinAtoms, numberMaxAtoms])

    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        quiet=True,
        share=False,
        server_port=args.port
    )
