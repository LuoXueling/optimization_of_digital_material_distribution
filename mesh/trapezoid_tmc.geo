// Gmsh project created on Thu Oct 23 12:28:50 2025
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {1.5, 1, 0, 1.0};
//+
Point(4) = {0.5, 1, 0, 1.0};
//+
Point(8) = {0.5+0.0618035, 1, 0, 1.0};
//+
Point(9) = {0.5-0.027639, 1-0.1+0.04472, 0, 1.0};
//+
Point(10) = {0.5+0.0618035, 1-0.1, 0, 1.0};
//+
Point(11) = {1.5-0.0618035, 1, 0, 1.0};
//+
Point(12) = {1.5+0.027639, 1-0.1+0.04472, 0, 1.0};
//+
Point(13) = {1.5-0.0618035, 1-0.1, 0, 1.0};

//+
Point(5) = {2.1, 1-0.1, 0, 1.0};
//+
Point(6) = {-0.1, 1.5895-0.1, 0, 1.0};
//+
Point(14) = {-0.1, 0, 0, 1.0};
//+
Point(15) = {2.1, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 12};
//+
Line(3) = {1, 9};
//+
Line(4) = {8, 11};
//+
Circle(5) = {9, 10, 8};
//+
Circle(6) = {12, 13, 11};
//+
Line(7) = {2, 15};
//+
Line(10) = {1, 14};
//+
Line(11) = {6, 14};
//+
Line(12) = {6, 5};
//+
Line(13) = {15, 5};

//+
Curve Loop(1) = {12, -13, -7, 2, 6, -4, -5, -3, 10, -11};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {3, 5, 4, -6, -2, -1};
//+
Plane Surface(2) = {2};
//+
Physical Surface("main", 15) = {2};
//+
Physical Surface("third_medium", 16) = {1};
//+
Physical Curve("top", 17) = {12};
//+
Physical Curve("bottom", 18) = {1};
//+
Physical Curve("third_medium_bottom", 21) = {10, 7};
//+
Physical Curve("left", 19) = {11};
//+
Physical Curve("right", 20) = {13};
//+
Transfinite Curve {10, 7} = 5 Using Progression 1;
//+
Transfinite Curve {13} = 50 Using Progression 1;
//+
Transfinite Curve {11} = 20 Using Progression 1;
//+
Transfinite Curve {12} = 100 Using Progression 1;
//+
Transfinite Curve {4} = 40 Using Progression 1;
//+
Transfinite Curve {2} = 50 Using Progression 1;
//+
Transfinite Curve {1} = 80 Using Progression 1.02;
//+
Transfinite Curve {3} = 60 Using Progression 1.01;
//+
Transfinite Curve {5} = 7 Using Progression 1;
//+
Transfinite Curve {6} = 7 Using Progression 1;
