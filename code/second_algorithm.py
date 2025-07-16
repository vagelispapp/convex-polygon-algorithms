from math import sqrt, cos, sin, asin, degrees, pi
import matplotlib.pyplot as plt

ERROR = 1e-5


# error handling
def greater_equal(a, b):
    return True if a >= b - ERROR else False


def equal(a, b):
    return True if a >= b - ERROR and a <= b + ERROR else False


def less_equal(a, b):
    return True if a <= b + ERROR else False


# vriskoume tin pio akraia koryfi ston axona x etsi wste
# elegxontas tis geitonikes koryfes na min einai se mia eythia.
# sti sunexeia ypologizoume tin orizousa twn koryfwn
# (de mporei na einai 0 logo toy parapanw sylogismoy)
# kai an exei thetiko prosimo einai counter-clockwise, alliws einai clockwize
def clockwise_order(polygon, n):
    farthest_vertex = polygon.index(max(polygon))
    a = polygon[farthest_vertex - 1]
    b = polygon[farthest_vertex]
    c = polygon[advance(farthest_vertex, n)]

    # ypologizoume tin orizousa
    if triangle_area(a, b, c) < 0:
        return polygon
    # epistrefoume tin lista twn koryfwn me tin anapodi seira
    return polygon[::-1]


# h(p)
# apostasi simeiou apo mia eutheia
# opou i eutheia dinetai san duo simeia
def distance_from_line(line, point):
    x1, y1 = line
    x = abs(
        (y1[0] - x1[0]) * (x1[1] - point[1]) - ((x1[0] - point[0]) * (y1[1] - x1[1]))
    )
    if equal(x, 0):
        x = 0
    y = sqrt((y1[0] - x1[0]) * (y1[0] - x1[0]) + (y1[1] - x1[1]) * (y1[1] - x1[1]))
    return x / y if y != 0 else 0


# ypologizoyme thn provoli enos simeiou se eytheia
def closest_point_on_line_from_point(line_A, line_B, point_P):
    x0, y0 = point_P
    x1, y1 = line_A
    x2, y2 = line_B

    # line coefficients
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    distance = (a * x0 + b * y0 + c) / (a * a + b * b) if (a * a + b * b) else 0

    x = x0 - distance * a
    y = y0 - distance * b

    return [x, y]


# gwnia metaksi dyo eytheiwn
def angle(opposite, hypotenuse):
    return asin(opposite / hypotenuse) if hypotenuse else 0


# elegxoume an mia gwnia einai amvleia, ypologizontas ean i provoli toy
# shmeiou P einai on_segment toy AB, ean einai tote exoyme okseia i orthi alliws amvleia
def check_if_angle_is_obtuse(line_A, line_B, point_P):
    closest_point = closest_point_on_line_from_point(
        line_A,
        line_B,
        point_P,
    )
    if on_segment(line_A, line_B, closest_point):
        return False
    return True


# vriskoyme to neo simeio x,y opou exei ginei rotate
# clockwise kata angle me vasi to arxiko
def rotate(fixed, point, angle):
    fixed_x, fixed_y = fixed
    point_x, point_y = point

    x = fixed_x + cos(angle) * (point_x - fixed_x) + sin(angle) * (point_y - fixed_y)
    y = fixed_y - sin(angle) * (point_x - fixed_x) + cos(angle) * (point_y - fixed_y)
    return [x, y]


# epomeno vertex
def advance(x, polygon_vertices):
    # -2 wste o epomenos toy teleutaiou na einai o prwtos
    return x + 1 if x <= polygon_vertices - 2 else 0


# epistrefoume kai to proigoymeno vertex
def line(start_point, polygon):
    return [polygon[start_point - 1], polygon[start_point]]


# ypologismos y(p) y = gamma mikro elliniko
# to opoio exei to diplasio h() apo to simeio p,
# vriskoume to symetriko toy simeiou tomis ths pleuras C
# me thn ekastote pleura me kentro to p
def y(center, point_a, point_b, polygon):
    # simeio tomis
    intersect_L1L2 = intersecting_lines(
        polygon[point_a],
        polygon[point_a - 1],
        polygon[point_b],
        polygon[point_b - 1],
    )
    # y(p)
    # an den yparxei simeio tomis epistrefoume tin koryfi a
    return (
        [
            2 * center[0] - intersect_L1L2[0],
            2 * center[1] - intersect_L1L2[1],
        ]
        if intersect_L1L2
        else polygon[point_a]
    )


# intersection point an den yparxei gyrnaei 0
# elegxoume an oi dyo pleures temnontai kai epistrefoume tin tomi tous
def intersecting_lines(line1_start, line1_end, line2_start, line2_end):
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return 0
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        denominator
    )
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        denominator
    )
    return [x, y]


# vriskoume tin parallili eytheia tis pleyras pou pernaei apo to point
# gia na ypologisoume tin pleura poy tha exei gnwsto to mid point
# ws parallili orizoume tin eytheia pou pernaei apo to point kai to second_point
def find_parallel_line(line_start, line_end, point):
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = point

    second_point_x = x3 + x2 - x1
    second_point_y = y3 + y2 - y1

    second_point = [second_point_x, second_point_y]

    return second_point


# elegxoume an i epomeni koryfi einai katw apo tin grammi pou orizoun
# ta simeia line_start, line_end xrisimopoiontas to cross product
def intersects_P_below_b(line_start, line_end, point):
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = point
    return (x2 - x1) * (y3 - y1) < (y2 - y1) * (x3 - x1)


# elegxoume an i epomeni koryfh einai panw apo tin grammi pou orizoun
# ta simeia line_start, line_end xrisimopoiontas to cross product
def intersects_P_above_b(line_start, line_end, point):
    x1, y1 = line_start
    x2, y2 = line_end
    x3, y3 = point
    return (x2 - x1) * (y3 - y1) > (y2 - y1) * (x3 - x1)


# vriskoume to midpoint
def midpoint(a, b):
    x1, y1 = a
    x2, y2 = b
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def distance_between_points(a, b):
    x1, y1 = a
    x2, y2 = b
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


# ypologizoume an ena simeio x einai anamesa apo tis koryfes tis grammis
# sygrinonts tis apostaseis toy simeiou apo tis akres, exontas ws dedomeno
# oti to simeio x einai panw sti grammi, dioti i grammi einai flush me to
# segment pou orizoun oi dyo koryfes
# logo toy floating point tis python den ginotan na ypologisoume swsta
# tin isotita (den htan pote ises para mono se akeraious), etsi xrisimopoiw
# tin equal pou ypologizei tin isotita +- ena error (1e-5)
def on_segment(line_start, line_end, point):
    return (
        True
        if equal(
            distance_between_points(line_start, point)
            + distance_between_points(point, line_end),
            distance_between_points(line_start, line_end),
        )
        else False
    )


# ypologizoume tis gwnies gia to case 1 kai to trito event
def theta_angle(a, b, c, x, alpha, beta, gamma, case, n, polygon):
    # elegxoume to case gia na gia na vroume se poia periptwsi eimaste
    # '1' gia 2 flush legs '0' gia 1 flush leg
    if case:
        # gia tin gwnia theta_a ypologizoume tis apostaseis toy x apo tis alles dyo
        # kai dexomaste tin mikroteri apo autes, etsi wste na min xrisimopoiisoume thn ypotinousa

        theta_a = angle(
            distance_from_line([x, midpoint(alpha, gamma)], polygon[a]),
            distance_between_points(polygon[a], x),
        )
    else:
        # vriskoume to symetriko shmeio toy mid point [a-1] ws pros to a, esti
        # wste na eksasfalisoume oti tha einai sthn parallili ws pros tin B
        # pleura, kathws kai oti tha synexisei na einai mid point
        rightmost_gamma = [
            2 * x[0] - polygon[a][0],
            2 * x[1] - polygon[a][1],
        ]

        # hypotenuse = apostasi apo to [a-1] sto rightmost_gamma
        # opposite = apostasi apo to rightmost_gamma sto AC
        theta_a = angle(
            distance_from_line([x, gamma], rightmost_gamma),
            distance_between_points(rightmost_gamma, x),
        )

    if check_if_angle_is_obtuse(x, midpoint(alpha, gamma), polygon[a]):
        theta_a = pi - theta_a

    # hypotenuse = apostasi apo to [a-1] sto midpoint toy AB
    # opposite = apostasi midpoint toy AB apo to [a-1]->b
    theta_b = angle(
        distance_from_line([x, polygon[b]], midpoint(alpha, beta)),
        distance_between_points(x, midpoint(alpha, beta)),
    )

    if check_if_angle_is_obtuse(x, midpoint(alpha, beta), polygon[b]):
        theta_b = pi - theta_b

    # to theta_c tha einai panta 0 logo tis symbasis poy exoyme kanei
    # gia to oti i pleyra C tha einai panta flush me mia akmi toy P
    return theta_a, theta_b, 0


# ypologizoume to emvado toy trigwnoy
def triangle_area(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    return (1 / 2) * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def advance_b_to_right_chain(b, c, polygon, n):
    while distance_from_line(
        line(c, polygon),
        polygon[advance(b, n)],
    ) >= distance_from_line(line(c, polygon), polygon[b]):
        b = advance(b, n)
    return b


def move_a_if_low_and_b_if_high(
    a,
    b,
    c,
    ya,
    polygon,
    n,
):
    while distance_from_line(line(c, polygon), polygon[b]) > distance_from_line(
        line(c, polygon), polygon[a]
    ):
        if intersects_P_below_b(polygon[b], ya, polygon[advance(b, n)]):
            b = advance(b, n)
        else:
            a = advance(a, n)
            ya = y(polygon[a], a, c, polygon)
    return a, b, ya


def search_for_B_tangency(
    a,
    b,
    c,
    yb,
    alpha,
    beta,
    polygon,
    n,
):
    while intersects_P_below_b(
        polygon[b], yb, polygon[advance(b, n)]
    ) and distance_from_line(line(c, polygon), polygon[b]) >= distance_from_line(
        line(c, polygon), polygon[a - 1]
    ):
        b = advance(b, n)
        yb = y(polygon[a - 1], a, c, polygon)

    # se periptwsi poy den mpei stin if, gia na kanoume track to allo point toy a
    alpha = a
    # se periptwsi poy den mpei stin if gia to mid point, gia na gnwrizoume an prepei na perasei apo to mid point
    mid_alpha = 0
    # ena flag wste sth synexeia na mporoyme eukola na katalavoyme an to trigwno exei thn pleura B flush
    flush_beta = 0

    if intersects_P_above_b(
        polygon[b], yb, polygon[advance(b, n)]
    ) or distance_from_line(line(c, polygon), polygon[b]) < distance_from_line(
        line(c, polygon), polygon[a - 1]
    ):
        # kanoume flush to side B me to b, b-1 kai enimerwnoume to flag
        flush_beta = 1
        beta = [polygon[b - 1], b]
        mid = midpoint(polygon[b], polygon[b - 1])
        if distance_from_line(line(c, polygon), mid) < distance_from_line(
            line(c, polygon), polygon[a - 1]
        ):
            mid_alpha = a - 1
    else:
        beta = [yb, b]
    return alpha, mid_alpha, b, beta, yb, flush_beta


def triangle_points(b, c, alpha, beta, mid_alpha, polygon):
    beta = intersecting_lines(
        beta[0],
        polygon[beta[1]],
        polygon[c],
        polygon[c - 1],
    )

    # se periptwsi poy de theloyme i pleyra A na exei
    # meso to a-1 to mid_alpha tha exei timi 0
    if mid_alpha:
        # ypologizoume to apenanti diagonio simeio toy shmeioy B se sxesi me to midpoint
        # kai sti synexeia ypologizoume to alpha, xrisimopoiontas tin parallili grammi ths apenanti
        # pleuras pou pernaei apo ayto to simeio. To alpha einai to simeio tomis tis parallilis grammis
        # me tin B
        diagonal_point = [
            2 * polygon[mid_alpha][0] - beta[0],
            2 * polygon[mid_alpha][1] - beta[1],
        ]

        # ypologizoume to deutero simeio wste na vroume thn parallili grammi poy pernaei apo to diagonal_point
        second_point_for_parallel_line = find_parallel_line(
            polygon[c], polygon[c - 1], diagonal_point
        )

        alpha = intersecting_lines(
            diagonal_point,
            second_point_for_parallel_line,
            polygon[b],
            polygon[b - 1],
        )

        gamma = intersecting_lines(
            alpha,
            polygon[mid_alpha],
            polygon[c],
            polygon[c - 1],
        )
    else:
        gamma = intersecting_lines(
            polygon[alpha],
            polygon[alpha - 1],
            polygon[c],
            polygon[c - 1],
        )

        alpha = intersecting_lines(
            polygon[alpha],
            polygon[alpha - 1],
            polygon[b],
            polygon[b - 1],
        )
    return alpha, beta, gamma


def inner_triangle_base_of_theta(
    a, b, c, alpha, beta, gamma, x, theta, case, n, polygon, name
):
    # to C_star einai i tomi tis pleuras A me tin nea C pou einai stramenei kata angle clockwize
    # to triangle einai mia lista tis morfis (area, vertex_a, vertex_b, vertex_c)
    if name == "a" or name == "b":
        # gia na elegksoume mono to theta pou exei kalesei tin function
        # apo to locals theloume to name to opoio tha einai a i b
        # an theloume to theta_a tote to opposite vertex tha einai to b kai anapoda
        name_value = locals()[name]
        opposite_vertex = locals()["b"] if name == "a" else locals()["a"]

        triangle = validation(
            polygon[name_value],
            x,
            polygon[opposite_vertex],
            polygon[opposite_vertex - 1],
            c,
            name,
            polygon,
        )

        triangle2 = validation(
            polygon[name_value],
            rotate(polygon[name_value], x, theta),
            polygon[opposite_vertex],
            polygon[opposite_vertex - 1],
            c,
            name,
            polygon,
        )

        # plt.waitforbuttonpress()
        return max(triangle, triangle2)
    else:
        C_star_validation = [
            alpha,
            polygon[a - 1],
            polygon[c],
            rotate(polygon[c], polygon[c - 1], theta),
        ]
        if not valid(*C_star_validation):
            return [-1, None, None]

        C_star = intersecting_lines(*C_star_validation)
        print("C_star: ", C_star)
        C_star_A_midpoint = midpoint(C_star, alpha)

        B_star_validation = [
            polygon[b],
            polygon[b - 1],
            polygon[c],
            rotate(polygon[c], polygon[c - 1], theta),
        ]
        if not valid(*B_star_validation):
            return [-1, None, None]
        # to B_star einai i tomi tis pleuras B me tin nea C pou einai stramenei kata angle clockwize
        B_star = intersecting_lines(*B_star_validation)

        B_star_A_midpoint = midpoint(B_star, alpha)

        print(f"theta_{name}")
        area = draw_triangle(C_star_A_midpoint, B_star_A_midpoint, polygon[c], polygon)
        triangle = [area, C_star_A_midpoint, B_star_A_midpoint]
    return triangle


# elegxoume an einai valid ena simeio kai an einai epistrefoume tis dyo
# koryfes toy trigwnoy se lista tis morfis (area, chord_a, chord_b)
# an den einai valid epistrefoume -1 gia to area kai None gia ta simeia
def validation(
    intersecting_line_start,
    intersecting_line_end,
    intersected_line_start,
    intersected_line_end,
    c,
    name,
    polygon,
):
    validation_parameters = [
        intersecting_line_start,
        intersecting_line_end,
        intersected_line_start,
        intersected_line_end,
    ]
    if not valid(*validation_parameters):
        return [-1, None, None]

    inner_x = intersecting_lines(*validation_parameters)
    print(f"inner_{name}_case1")
    # plt.waitforbuttonpress()

    # to trigwno pou sximatizetai apo to simeio tomis tis pleyras
    # area = draw_triangle(inner_x, intersecting_line_start, polygon[c], polygon)

    # afoy to trigwno exei perasei to validation pernoyme to pio akraio wrologiaka
    # simeio tis pleyras toy P tin opoia temnei (intersected_line_start)
    area = draw_triangle(
        intersected_line_start, intersecting_line_start, polygon[c], polygon
    )
    return [area, inner_x, intersecting_line_start]


# gia na theoritei ena simeio valid tha prepei na vriskete
# panw se mia pleura tou P i se mia koryfi
def valid(
    intersecting_line_start,
    intersecting_line_end,
    intersected_line_start,
    intersected_line_end,
):
    point_x = intersecting_lines(
        intersecting_line_start,
        intersecting_line_end,
        intersected_line_start,
        intersected_line_end,
    )
    if not point_x:
        return False
    if on_segment(intersected_line_start, intersected_line_end, point_x):
        return True
    return False


# an i parallili me to chord pou pernaei apo to trito simeio temnei to P
# to trigwno mporei na megalwsei kai allo afksanontas to trito simeio
# kata mia thesi clockwise, proypothesi to chord na dinetai clockwise
def check_chord_parallel(point, chord_a, chord_b, n, polygon):
    parallel_to_cord = find_parallel_line(
        polygon[chord_a], polygon[chord_b], polygon[point]
    )
    if intersects_P_below_b(
        polygon[point], parallel_to_cord, polygon[advance(point, n)]
    ):
        return polygon[advance(point, n)]
    return polygon[point]


# two_flush_legs case
def two_flush_legs(a, b, c, alpha, beta, gamma, x, case, n, polygon):
    # calculate theta angles
    theta_a, theta_b, theta_c = theta_angle(
        a, b, c, x, alpha, beta, gamma, case, n, polygon
    )

    # a(theta) = a'
    inner_triangle_area_theta_a = inner_triangle_base_of_theta(
        a, b, c, alpha, beta, gamma, x, theta_a, case, n, polygon, "a"
    )

    # b(theta) = b'
    inner_triangle_area_theta_b = inner_triangle_base_of_theta(
        a, b, c, alpha, beta, gamma, x, theta_b, case, n, polygon, "b"
    )

    min_theta = min(theta_a, theta_b, theta_c)
    print(
        degrees(theta_a),
        theta_a,
        "-----",
        degrees(theta_b),
        theta_b,
        "-----",
        degrees(theta_c),
        theta_c,
    )

    # plt.waitforbuttonpress()
    inner_triangle_area_theta_min = inner_triangle_base_of_theta(
        a, b, c, alpha, beta, gamma, x, min_theta, case, n, polygon, "min_theta"
    )

    # epistrefoume to megisto trigwno apo ta parapanw
    inner_triangle_area, inner_a, inner_b = max(
        inner_triangle_area_theta_min,
        inner_triangle_area_theta_a,
        inner_triangle_area_theta_b,
    )
    return inner_triangle_area, inner_a, inner_b


# gia na theorithei local minimum ena trigwno tha prepei
# ta midpoints kathe pleuras na vriskontai sta segments toy
# trigwnou pou efaptontai me to P
def local_minimum(alpha, beta, gamma, a, b, c, polygon):
    midpoint_alpha = midpoint(alpha, gamma)
    midpoint_beta = midpoint(beta, alpha)
    midpoint_gamma = midpoint(gamma, beta)

    draw_midpoint(midpoint_alpha, "A")
    draw_midpoint(midpoint_beta, "B")
    draw_midpoint(midpoint_gamma, "C")

    # an estw kai ena den plirei tis proypotheseis tote den einai local minimum
    if not on_segment(polygon[a], polygon[a - 1], midpoint_alpha):
        return False
    if not on_segment(polygon[b], polygon[b - 1], midpoint_beta):
        return False
    if not on_segment(polygon[c], polygon[c - 1], midpoint_gamma):
        return False
    return True


def global_maximum_area(a, b, c, global_maximum, inner_triangle_area):
    if not global_maximum or inner_triangle_area > global_maximum[0]:
        return [inner_triangle_area, a, b, c]
    return global_maximum


def maximal_enclosed_triangle(
    a, b, c, alpha, beta, gamma, mid_alpha, flush_b, n, global_maximum, polygon
):
    # first inner triangle
    # ypologizetai pairnontas tin epomeni clockwise koryfi apo ta mid points twn
    # [a,a-1] kai [b,b-1] i opoia einai i a kai b antistoixa
    print("first inner triangle")
    first_inner_a = check_chord_parallel(a, b, c, n, polygon)

    first_inner_b = check_chord_parallel(b, c, a, n, polygon)

    first_inner_c = polygon[c]

    inner_triangle_area = draw_triangle(
        first_inner_a, first_inner_b, first_inner_c, polygon
    )

    global_maximum = global_maximum_area(
        first_inner_a, first_inner_b, polygon[c], global_maximum, inner_triangle_area
    )

    # elegxoume an to mid_alpha kai to flush_b kai an einai 0 kai 1 antistoixa, to trigwno exei 2 flush legs
    # afoy i pleura A de xriazetai na pernaei apo to mid_alpha, epomenws einai flush me to [a,a-1] kai
    # i pleura B einai flush me to [b,b-1], kathos tin exoume orisei etsi
    if not mid_alpha and flush_b:
        # --------------------------two flush legs--------------------------
        print("2 flush legs")
        # flag gia to case '2 flush legs'
        case = 1
        x = midpoint(alpha, polygon[c])
        print("x:", x)
        x_x, x_y = x
        c_line_x, c_line_y = line(c, polygon)
        plt.plot(x_x, x_y, marker="X", color="r", label=f"x {x_x, x_y}")
        plt.axline(c_line_x, c_line_y, linewidth=1, color="r")
        plt.axline(alpha, gamma, linewidth=1, color="k")
        plt.axline(beta, alpha, linewidth=1, color="g")
        # plt.waitforbuttonpress()
        (
            inner_triangle_area,
            inner_a,
            inner_b,
        ) = two_flush_legs(a, b, c, alpha, beta, gamma, x, case, n, polygon)
    else:
        print("1 flush leg")
        # flag gia to case '1 flush leg'
        case = 0
        (
            inner_triangle_area,
            inner_a,
            inner_b,
        ) = two_flush_legs(
            a, b, c, alpha, beta, gamma, polygon[a - 1], case, n, polygon
        )
        print(alpha, beta, gamma)
        c_line_x, c_line_y = line(c, polygon)
        plt.axline(c_line_x, c_line_y, linewidth=1, color="r")
        plt.axline(alpha, gamma, linewidth=1, color="k")
        plt.axline(beta, alpha, linewidth=1, color="g")
        # plt.waitforbuttonpress()
    global_maximum = global_maximum_area(
        inner_a,
        inner_b,
        polygon[c],
        global_maximum,
        inner_triangle_area,
    )
    return global_maximum


def minimal_enclosing_triangles(a, b, n, polygon):
    alpha = 0
    beta = 0
    gamma = 0
    global_minimum = None
    global_maximum = None

    for c in range(n):
        clear_plot(a, b, c, polygon)

        animation_step("Advance b to right chain")

        # Advance b to right chain
        b = advance_b_to_right_chain(b, c, polygon, n)

        draw_figure(a, b, c, polygon)
        c_line_x, c_line_y = line(c, polygon)

        animation_step(" Draw line C")
        plt.axline(c_line_x, c_line_y, linewidth=2, color="r")

        intersection_BC = intersecting_lines(
            polygon[b],
            polygon[b - 1],
            polygon[c],
            polygon[c - 1],
        )

        # elegxoume to intersection point tis pleyras B kai C
        # an den yparxei (intersection_BC == 0) de synexizw ton ypologizmo to trigwnoy
        # afoy oi dyo pleures einai paralliles
        if not intersection_BC:
            b_line_x, b_line_y = line(b, polygon)
            plt.axline(b_line_x, b_line_y, linewidth=2, color="r")
            animation_step("There is no local triangle.\n Sides B and C are parallel")
            continue

        # Move a if low, and b if high
        # gamma a
        ya = y(polygon[a], a, c, polygon)

        animation_step("Calculate ya")
        draw_figure(a, b, c, polygon, ya, ya_options=["c", "ya"])

        a, b, ya = move_a_if_low_and_b_if_high(
            a,
            b,
            c,
            ya,
            polygon,
            n,
        )
        animation_step("Move a if low and b if high")

        draw_figure(a, b, c, polygon, ya, ya_options=["c", "ya"])

        # Search for the B tangency
        # gamma b
        yb = y(polygon[b], b, c, polygon)

        animation_step("Calculate yb")

        draw_figure(
            a,
            b,
            c,
            polygon,
            ya,
            yb,
            ya_options=["c", "ya"],
            yb_options=["k", "yb"],
        )

        alpha, mid_alpha, b, beta, yb, flush_b = search_for_B_tangency(
            a, b, c, yb, alpha, beta, polygon, n
        )

        animation_step("Search for the B tangency")

        draw_figure(
            a,
            b,
            c,
            polygon,
            ya,
            yb,
            ya_options=["c", "ya"],
            yb_options=["k", "yb"],
        )

        alpha, beta, gamma = triangle_points(b, c, alpha, beta, mid_alpha, polygon)

        print(abs(triangle_area(alpha, beta, gamma)))

        area = draw_triangle(alpha, beta, gamma, polygon)
        if local_minimum(alpha, beta, gamma, a, b, c, polygon):
            add_suptitle("Triangle is local minimum")
            if not global_minimum or area < global_minimum[0]:
                global_minimum = [area, alpha, beta, gamma]
        else:
            # TODO: check
            # ?? ti ginetai me ta P pou den exoun global minimum
            if not global_minimum or area < global_minimum[0]:
                global_minimum = [area, alpha, beta, gamma]
            add_suptitle("Triangle is not local minimum")

        # --------------------------maximal enclosed triangle--------------------------
        global_maximum = maximal_enclosed_triangle(
            a, b, c, alpha, beta, gamma, mid_alpha, flush_b, n, global_maximum, polygon
        )
        # --------------------------maximal enclosed triangle--------------------------

    if global_minimum:
        alpha, beta, gamma = global_minimum[1:]
        draw_triangle(alpha, beta, gamma, polygon)
        add_suptitle(f"This is the global minimum for this polygon")

    if global_maximum:
        inner_a, inner_b, inner_c = global_maximum[1:]
        plt.figure(figsize=(8, 8), layout="constrained", dpi=100)

        draw_triangle(inner_a, inner_b, inner_c, polygon)
        add_suptitle(f"This is the global maximum inner triangle for this polygon")

    return global_minimum, global_maximum


# -----------------------------------------------------------------------------
# -------------------------------drawing polygon-------------------------------
# -----------------------------------------------------------------------------


# katharizw to plot, gia tin nea C
def clear_plot(a, b, c, polygon):
    plt.clf()
    draw_polygon(polygon)

    a_x, a_y = polygon[a]
    b_x, b_y = polygon[b]
    c_x, c_y = polygon[c]
    plt.axis("equal")
    plt.plot(a_x, a_y, marker="o", color="m", label=f"a {a_x, a_y}")
    plt.plot(b_x, b_y, marker="o", color="y", label=f"b {b_x, b_y}")
    plt.plot(c_x, c_y, marker="o", color="r", label=f"c {c_x, c_y}")
    plt.legend()


def final_plot(a, b, c, polygon):
    plt.clf()
    draw_polygon(polygon)

    a_x, a_y = a
    b_x, b_y = b
    c_x, c_y = c
    plt.axis("equal")
    plt.plot(a_x, a_y, marker="o", color="k", label=f"A {a_x, a_y}")
    plt.plot(b_x, b_y, marker="o", color="g", label=f"B {b_x, b_y}")
    plt.plot(c_x, c_y, marker="o", color="r", label=f"C {c_x, c_y}")
    plt.legend()


# gia na fainetai vima vima i diadikasia
def animation_step(title):
    print(title)
    plt.title(title)
    plt.waitforbuttonpress()


def add_suptitle(title):
    print(title)
    plt.suptitle(title)
    # plt.waitforbuttonpress()


def draw_polygon(polygon):
    x, y = zip(*polygon)
    # gia na 'kleisei' to polygwno
    x_last_side, y_last_side = zip(*[polygon[0], polygon[-1]])
    plt.axis("equal")
    plt.plot(x, y, x_last_side, y_last_side, linestyle="-", color="b")


def draw_midpoint(midpoint, name):
    x, y = midpoint
    add_suptitle(f"midpoint of side {name}")
    plt.plot(x, y, "x", ms=10, color="m")
    plt.legend()
    # plt.waitforbuttonpress()


def draw_figure(a, b, c, polygon, *vertices, **options):
    clear_plot(a, b, c, polygon)
    c_line_x, c_line_y = line(c, polygon)
    for index, vertex in enumerate(vertices):
        vetrex_x, vertex_y = vertex
        plt.axis("equal")
        plt.plot(
            vetrex_x,
            vertex_y,
            marker="o",
            color=list(options.values())[index][0],
            label=f"{list(options.values())[index][1]} {vetrex_x, vertex_y}",
        )
    plt.axline(c_line_x, c_line_y, linewidth=2, color="r")
    plt.legend()


def draw_triangle(alpha, beta, gamma, polygon):
    alpha_x, alpha_y = alpha
    beta_x, beta_y = beta
    gamma_x, gamma_y = gamma

    plt.waitforbuttonpress()
    final_plot(alpha, beta, gamma, polygon)

    plt.axis("equal")

    plt.plot([alpha_x, gamma_x], [alpha_y, gamma_y], color="k", label="line A")
    plt.plot([alpha_x, beta_x], [alpha_y, beta_y], color="g", label="line B")
    plt.plot([gamma_x, beta_x], [gamma_y, beta_y], color="r", label="line C")
    plt.legend()

    area = abs(triangle_area(alpha, beta, gamma))
    plt.title(f"area:{area}", loc="center")
    # plt.waitforbuttonpress()
    return area


# -----------------------------------------------------------------------------
# -------------------------------drawing polygon-------------------------------
# -----------------------------------------------------------------------------


def main():
    # ena arxeio tis morfis [[...],[...],...,[...]] me tis sintetagmentes twn
    # polygwnwn pou theloyme ma vroyme ta megista kai elaxista trigwna
    with open("/Users/Vp/Desktop/polygon.txt", "r") as f:
        contents = f.read()
        # katharizoyme ta contents apo ta "[" , "]"
        contents = contents.replace("[", " ")
        contents = contents.replace("]", " ")
        # xwrizoume se lines ta contents, kathe line exei apo ena polygwno
        lines = contents.split("\n")
        polygons = []
        for line in lines:
            # se periptosi pou exei ksexastei enas kenos xaraktiras
            # sto telos i se kapoio simeio toy txt arxeioy
            if not line:
                continue
            coordinates = line.split(",")
            polygon = []
            # ana dyo prosthetoyme ta coordinates toy line se mia lista
            for index, _ in enumerate(coordinates):
                if index % 2 != 0:
                    continue
                polygon.append(
                    [float(coordinates[index]), float(coordinates[index + 1])]
                )
            polygons.append(polygon)

    # gia kathe ena polygwno ypologizoyme min kai max triangles
    for polygon in polygons:
        # constrained gia na min yparxoun polla kena sto figure
        print(polygon)
        plt.figure(figsize=(8, 8), layout="constrained", dpi=100)
        n = len(polygon)

        a = 1
        b = 2

        # se periptwsi pou den dwthoun se wrologiaki fora oi koryfes
        # allazw 'fora' etsi wste na einai
        polygon = clockwise_order(polygon, n)

        global_minimum, global_maximum = minimal_enclosing_triangles(a, b, n, polygon)

        # ta apotelesmata ta pername se ena arxeio results.txt
        with open("/Users/Vp/Desktop/results.txt", "w") as f:
            f.write(
                f"For the polygon {polygon}, this is the global minimum triangle{global_minimum[1:]} with area {global_minimum[0]} \
                \nand this is the global maximum triangle {global_maximum[1:]} with area {global_maximum[0]} \n\n"
            )

    plt.show()


if __name__ == "__main__":
    main()
