🧠 Thesis: Enclosing and Inscribed Triangles in Convex Polygons

This thesis explores the problem of identifying triangles with minimum and maximum area that are either enclosing or inscribed within a given convex polygon — a problem with significant applications in fields such as robotics and collision detection. These applications often require simplifying shapes to those with the fewest possible sides, which is the central goal of this study.

📌 What Was Implemented

Two algorithms were developed:

Minimal Enclosing TrianglesCalculates all locally minimal-area triangles that enclose a convex polygon and identifies the global minimum.

Maximal Inscribed TriangleA sequential algorithm that builds upon the first to compute the triangle with the maximum possible area that fits entirely inside the polygon.

In both cases, area is the primary optimization criterion.

💾 Technologies Used

Language: Python

Visualization: Matplotlib

Math Operations: Python's built-in math module (sqrt, cos, sin, asin, degrees, pi)

📊 Features

Full implementation of both algorithms

Visualizations of their execution and output

Reproducible and modular Python code

🧑‍🏫 Author & Info

Name: Evangelos Pappas

Institution: University of Ioannina, School of Engineering

Department: Computer Science & Engineering

Year: 2023
