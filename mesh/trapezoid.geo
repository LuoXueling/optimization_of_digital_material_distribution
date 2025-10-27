// Gmsh project created on Fri Oct 17 14:44:06 2025
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {1.5, 1, 0, 1.0};
//+
Point(4) = {0.5, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Transfinite Curve {3} = 50 Using Bump 0.1;
//+
Transfinite Curve {4} = 50 Using Bump 0.05;
//+
Transfinite Curve {2} = 50 Using Bump 0.05;
//+
Transfinite Curve {1} = 50 Using Bump 0.1;
//+
Physical Curve("top", 5) = {3};
//+
Physical Curve("bottom", 6) = {1};
//+
Physical Surface("main", 7) = {1};
