import * as sim from "lib-simulation-wasm";

const simulation = new sim.Simulation();

// get the canvas ID
const viewport = document.getElementById("viewport");

// dimensiomns of viewport (canvas)
const viewport_width = viewport.width;
const viewport_height = viewport.height;

const viewportScale = window.devicePixelRatio || 1;

viewport.width = viewport_width * viewportScale;
viewport.height = viewport_height * viewportScale;

viewport.style.width = viewport_width + 'px';
viewport.style.height = viewport_height + 'px';

//canvas rendering context 2d
const ctxt = viewport.getContext("2d");

ctxt.scale(viewportScale, viewportScale);

// console.log(viewport_width, viewport_height);

// function to draw a triangle on canvas
CanvasRenderingContext2D.prototype.drawTriangle =
    function (x, y, size, rotation) {
        this.beginPath();

        this.moveTo(
            x - Math.sin(rotation) * size * 1.5,
            y + Math.cos(rotation) * size * 1.5,
        );

        this.lineTo(
            x - Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
            y + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x - Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
            y + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
        );

        this.lineTo(
            x - Math.sin(rotation) * size * 1.5,
            y + Math.cos(rotation) * size * 1.5,
        );

        this.stroke();
        this.strokeStyle = 'rgb(0,0,0)';
        this.fillStyle = '#EFEFEF';
        this.fill();
    };


// function to draw a circle on canvas
CanvasRenderingContext2D.prototype.drawCircle =
    function (x, y, radius) {
        this.beginPath();

        // ---
        // | Circle's center.
        // ----- v -v
        this.arc(x, y, radius, 0, 2.0 * Math.PI);
        // ------------------- ^ -^-----------^
        // | Range at which the circle starts and ends, in radians.
        // |
        // | By manipulating these two parameters you can e.g. draw
        // | only half of a circle, Pac-Man style.
        // ---

        this.fillStyle = 'rgb(0, 255, 128)';
        this.fill();
    };

function redraw() {
    ctxt.clearRect(0, 0, viewport_width, viewport_height);

    simulation.step();

    const world = simulation.world();

    for (const food of world.foods) {
        ctxt.drawCircle(
            food.x * viewport_width,
            food.y * viewport_height,
            (0.01 / 2.0) * viewport_width,
        );
    }

    for (const animal of world.animals) {
        ctxt.drawTriangle(
            animal.x * viewport_width,
            animal.y * viewport_height,
            0.01 * viewport_width,
            animal.rotation,
        );
    }

    // requestAnimationFrame() schedules code only for the next frame.
    //
    // Because we want for our simulation to continue forever, we've
    // gotta keep re-scheduling our function:
    requestAnimationFrame(redraw);
}

redraw();