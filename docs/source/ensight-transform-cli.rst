``ensight_transform`` CLI tool
==============================

Though the library has explicit focus on reading the EnSight Gold format,
as opposed to creating cases from scratch or modifying them, by its nature
it can be used to modify arrays used in the binary files (node coordinates,
variable data, etc.).

At the moment, there is one ready-made script available: ``ensight_transform``
can be used to translate, scale, rotate, or do any other affine transformation
to nodes in EnSight Gold case (for all parts, or only selected subset).

::

    usage: ensight_transform.py [-h] [--only-parts regex] [--translate dX dY dZ | --scale sX sY sZ | --matrix a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44] *.case

    ensight_transform script
    ========================

    This script does **in-place** transformation of node coordinates
    in given EnSight Gold case. Your original geofile will be modified!

    Examples:

        # increment X coordinate
        ensight_transform --translate 1 0 0 sphere.case

        # scale by 1000 (eg. m -> mm conversion)
        ensight_transform --scale 1e3 1e3 1e3 sphere.case

        # rotation matrix
        ensight_transform --matrix \
            0 -1  0  0 \
            1  0  0  0 \
            0  0  1  0 \
            0  0  0  1 \
            sphere.case

        # transform only "internalMesh" part
        ensight_transform --translate 1 0 0 --only-parts internalMesh motorbike.case

    For commandline usage, run the script with ``--help``.

    positional arguments:
      *.case                EnSight Gold case (C Binary)

    options:
      -h, --help            show this help message and exit
      --only-parts regex    only export parts matching given regular expression (Python re.search)
      --translate dX dY dZ  translate nodes by given dX, dY, dZ values
      --scale sX sY sZ      scale nodes by given sX, sY, sZ values
      --matrix a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44
                            do affine transformation of nodes by multiplying via the (a_ij) matrix
