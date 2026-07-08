# Multimodal Guardrail Competition

## Task

Your system must classify both the image and the prompt together. Some
examples look harmless from the text alone, while others need the visual
context to decide whether the request is safe.

Labels:

- `attack`: jailbreaks, prompt injections, or requests that try to elicit unsafe behavior.
- `benign`: ordinary image-grounded requests or harmless prompts.

The data can be downloaded from [here](https://drive.google.com/file/d/13Vt_xvqW7rYQkdCPFXCyttx2yp8mxNGq/view?usp=sharing)

## Starter code

`competition_skeleton.py`:

- loads `student/train.csv`;
- reads images with `dspy.Image`;
- configures the class Gemma multimodal models;
- defines a DSPy signature that predicts `attack` or `benign` from `image` + `prompt`;
- saves a compiled guardrail to `competition_guardrail/` using the full-program
  format.

## Submission

Your final submission is **your whole code plus the saved checkpoint
directory**. Submit:

- all of your code (e.g. your edited `competition_skeleton.py`, plus any
  other modules/signatures/optimizer scripts you wrote); and
- the `competition_guardrail/` directory produced by `.save(path,
  save_program=True)`, containing `program.pkl` and `metadata.json`.

```python
final_guardrail.save(OUTPUT_PATH, save_program=True)
```

Before submitting, sanity-check that the saved checkpoint has the same
performance as the compiled version:

```python
fresh = dspy.load("competition_guardrail", allow_pickle=True)
evaluate(fresh, valset)
```