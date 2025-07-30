let image_selector;
let image_selector_alt;
let run_model_button;
let form;

document.addEventListener("DOMContentLoaded", () => {
    image_selector = document.getElementById('image_selector');
    image_selector_alt = document.getElementById('image_selector_alt');
    run_model_button = document.getElementById('run_model_button');
    form = document.getElementById("form");

    form.reset()

    image_selector_alt.addEventListener("click", () => {
        image_selector.click();
    });

    image_selector.addEventListener("change", () => {
        image_selector_alt.innerText = image_selector.files[0].name;

        if (image_selector.files.length == 0)
        {
            run_model_button.style.display = 'none';
        } else {
            run_model_button.style.display = 'block';
        }
    });

    file_selected("Select File");
});

function file_selected(new_text) {
    image_selector_alt.innerText = new_text;

    if (image_selector.files.length == 0)
    {
        run_model_button.style.display = 'none';
    } else {
        run_model_button.style.display = 'block';
    }
}