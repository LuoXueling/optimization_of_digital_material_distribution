// Gmsh project created on Wed Nov 12 11:06:35 2025
//+
SetFactory("OpenCASCADE");
Sphere(1) = {0.75, 0.75, 0, 0.5, 0, Pi/2, 2*Pi};
//+
Point(3) = {0, 0, 0, 1.0};
//+
Point(4) = {1.5, 0, 0, 1.0};
//+
Point(5) = {1.5, 1.5, 0, 1.0};
//+
Point(6) = {0, 1.5, 0, 1.0};
//+
Point(7) = {0, 0, 0.75, 1.0};
//+
Point(8) = {1.5, 0, 0.75, 1.0};
//+
Point(9) = {1.5, 1.5, 0.75, 1.0};
//+
Point(10) = {0, 1.5, 0.75, 1.0};

//+
Line(4) = {6, 5};
//+
Line(5) = {5, 4};
//+
Line(6) = {4, 3};
//+
Line(7) = {3, 6};
//+
Line(8) = {6, 10};
//+
Line(9) = {10, 9};
//+
Line(10) = {9, 5};
//+
Line(11) = {9, 8};
//+
Line(12) = {8, 4};
//+
Line(13) = {8, 7};
//+
Line(14) = {7, 3};
//+
Line(15) = {7, 10};
//+
Curve Loop(3) = {13, 15, 9, 11};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {12, -5, -10, 11};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {9, 10, -4, 8};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {15, -8, -7, -14};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {13, 14, -6, -12};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {6, 7, 4, 5};
//+
Curve Loop(9) = {3};
//+
Plane Surface(8) = {8, 9};

//+
Surface Loop(2) = {3, 7, 6, 5, 4, 8, 1};
//+
Surface Loop(3) = {9, 4, 7, 3, 6, 5, 1};
//+
Volume(2) = {3};
//+
Physical Surface("top", 16) = {3};
//+
Physical Surface("third_medium_bottom", 17) = {9};
//+
Physical Surface("bottom", 18) = {2};
//+
Physical Surface("semisphere-top", 19) = {1};
//+
Transfinite Curve {8, 14, 12, 10} = 7 Using Progression 1;
//+
Transfinite Curve {15, 9, 11, 13, 5, 6, 7, 4} = 15 Using Progression 1;
//+
Transfinite Curve {2, 2} = 30 Using Progression 1;
//+
Transfinite Curve {3, 3, 3} = 100 Using Progression 1;
//+
Physical Volume("main", 20) = {1};
//+
Physical Volume("third_medium", 21) = {2};
