//+
SetFactory("OpenCASCADE");
//+
Point(1) = {0., 0, 0, 1.0};
//+
Point(2) = {0.75, 0, 0, 1.0};
//+
Point(3) = {0.75, 0.75, 0, 1.0};
//+
Point(4) = {0, 0.75, 0, 1.0};
//+
Point(5) = {0.75, 0., 0, 1.0};
//+
Point(6) = {0.75-0.35*1.41421356, 0., 0, 1.0};
//+
Point(7) = {0.75-0.35, 0.35, 0, 1.0};
//+
Point(8) = {0.75, 0.35*1.41421356, 0, 1.0};

//+
Line(1) = {1, 6};
//+
Line(2) = {6, 2};
//+
Line(3) = {2, 8};
//+
Line(4) = {8, 3};
//+
Line(5) = {3, 4};
//+
Line(6) = {4, 1};
//+
Circle(7) = {6, 2, 7};
//+
Circle(8) = {7, 2, 8};
//+
Curve Loop(1) = {6, 1, 7, 8, 4, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, -3, -2, 7};
//+
Plane Surface(2) = {2};
//+
Physical Curve("top", 9) = {5};
//+
Physical Curve("left", 10) = {6};
//+
Physical Curve("right", 11) = {4, 3};
//+
Physical Curve("bottom", 12) = {1, 2};
//+
Physical Curve("outer_circle_left", 13) = {7};
//+
Physical Curve("outer_circle_top", 14) = {8};
//+
Physical Surface("third_medium", 15) = {1};
//+
Physical Surface("main", 16) = {2};
//+
Transfinite Curve {7, 8} = 30 Using Progression 1;
//+
Transfinite Curve {6, 5} = 4 Using Progression 1;
//+
Transfinite Curve {2, 3} = 30 Using Progression 1;
