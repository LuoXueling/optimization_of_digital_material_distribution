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
Point(10) = {0.75-0.32*1.41421356, 0., 0, 1.0};
//+
Point(11) = {0.75-0.32, 0.32, 0, 1.0};
//+
Point(12) = {0.75, 0.32*1.41421356, 0, 1.0};
//+
Point(14) = {0.75-0.25*1.41421356, 0., 0, 1.0};
//+
Point(15) = {0.75, 0.25*1.41421356, 0, 1.0};

//+
Line(1) = {1, 6};
//+
Line(2) = {6, 10};
//+
Line(3) = {10, 14};
//+
Line(4) = {14, 2};
//+
Line(5) = {2, 15};
//+
Line(6) = {15, 12};
//+
Line(7) = {12, 8};
//+
Line(8) = {8, 3};
//+
Line(9) = {3, 4};
//+
Line(10) = {4, 1};
//+
Circle(11) = {6, 2, 7};
//+
Circle(12) = {7, 2, 8};
//+
Circle(13) = {10, 2, 11};
//+
Circle(14) = {11, 2, 12};
//+
Circle(15) = {14, 2, 15};
//+
Curve Loop(1) = {9, 10, 1, 11, 12, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {12, -7, -14, -13, -2, 11};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {13, 14, -6, -15, -3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {15, -5, -4};
//+
Plane Surface(4) = {4};
//+
Physical Curve("top", 16) = {9};
//+
Physical Curve("bottom", 17) = {1, 2, 3, 4};
//+
Physical Curve("left", 23) = {10};
//+
Physical Curve("right", 18) = {8, 7, 6, 5};
//+
Physical Curve("outer_circle_left", 19) = {11};
//+
Physical Curve("outer_circle_top", 20) = {12};
//+
Physical Surface("third_medium", 21) = {1, 3, 4};
//+
Physical Surface("main", 22) = {2};
//+
Transfinite Curve {11, 12, 14, 13} = 30 Using Progression 1;
//+
Transfinite Curve {10, 9} = 4 Using Progression 1;
//+
Transfinite Curve {15} = 10 Using Progression 1;
//+
Transfinite Curve {2, 7} = 10 Using Progression 1;

