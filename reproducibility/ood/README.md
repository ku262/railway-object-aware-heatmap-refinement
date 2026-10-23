# OOD object compositing

This portable script implements the experiment's object placement operations:
RGBA rotation (uniform -25 to 25 degrees), alpha-bounds cropping, resizing to
a requested component-relative bounding-box area, positional jitter, local
brightness matching (gain clipped to 0.65-1.05), and alpha compositing.
It exports the composite, binary foreground mask, YOLO box and transform metadata.
Pixels outside the foreground mask remain unchanged.

## Run

Install dependencies with `python -m pip install -r requirements.txt`.
Supply your own background, an RGBA object image and a component box in pixel coordinates:

```sh
python compose.py --background background.png --asset object.png --part-box 20 20 300 200 --area-fraction 0.05 --seed 1400 --output example_output
```

The output directory must not already exist. Example box coordinates must be
replaced with a valid component box for your image. Experiment target area
fractions were 0.02, 0.05 and 0.10. The seed is supplied per image.

This is a portable implementation using the final extracted RGBA foregrounds,
not a byte-identical replay of the original 500-image cohort: original private
backgrounds, background assignments and historical intermediate alpha repairs
are not distributed. It does not regenerate assets from the generative model.
No industrial images, real-object donor crops, model weights or feature caches
are included. Use source-disjoint held-out backgrounds for evaluation, not
backgrounds fitted by the evaluated model/reference bank.

## Licensing of object assets

This code contains no MiniMax model code or weights. The AI-generated object
assets used in the paper were produced with MiniMax H3 and are not included in
this repository because of the territorial restrictions in the MiniMax H3
Community License. The repository code license does not apply to those images.
