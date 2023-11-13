// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return all_gallery_buttons().find(elem => elem.classList.contains('selected')) ?? null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex(elem => elem.classList.contains('selected'));
}

function extract_image_from_gallery(gallery) {
    if (gallery.length == 0) {
        return [null];
    }
    if (gallery.length == 1) {
        return [gallery[0]];
    }

    var index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [gallery[index]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}
function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    let buttons = gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button');
    for (let i = 0; i < buttons.length; i++) {
        if (buttons[i].classList.contains('selected')) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    var res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    let res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

function create_submit_args(args) {
    var res = Array.from(args);

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if (Array.isArray(res[res.length - 3])) {
        res[res.length - 3] = null;
    }

    return res;
}

function showSubmitButtons(tabname, show) {
    gradioApp().getElementById(tabname + '_interrupt').style.display = show ? "none" : "block";
}

function showRestoreProgressButton(tabname, show) {
    var button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;

    button.style.display = show ? "flex" : "none";
}

function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

function localSet(k, v) {
    try {
        localStorage.setItem(k, v);
    } catch (e) {
        console.warn(`Failed to save ${k} to localStorage: ${e}`);
    }
}

function localGet(k, def) {
    try {
        return localStorage.getItem(k);
    } catch (e) {
        console.warn(`Failed to load ${k} from localStorage: ${e}`);
    }

    return def;
}

function localRemove(k) {
    try {
        return localStorage.removeItem(k);
    } catch (e) {
        console.warn(`Failed to remove ${k} from localStorage: ${e}`);
    }
}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function txt2img_submit() {
    showSubmitButtons('txt2img', false);

    var id = randomId();
    localSet("txt2img_task_id", id);

    var funProgress = function(id) {
        request('/internal/progress', {id_task: id, is_img2img: false}, function(res) {
            if (res.completed) {
                showSubmitButtons('txt2img', true);
                return;
            }
            if (res.interrupted) {
                showSubmitButtons('txt2img', true);
                return;
            }
            setTimeout(() => {
                funProgress(id)
            }, 500);
        }, function(error) {
            console.log(error)
            return
        })
    }
    
    funProgress(id)
    res = Array.from(arguments);
    res[0] = id
    return res.slice(0, res.length - 2)
}

function img2img_submit() {
    showSubmitButtons('img2img', false);

    var id = randomId();
    localSet("img2img_task_id", id);

    var funProgress = function(id) {
        request('/internal/progress', {id_task: id, is_img2img: true}, function(res) {
            if (res.completed) {
                showSubmitButtons('img2img', true);
                return;
            }
            if (res.interrupted) {
                showSubmitButtons('img2img', true);
                return;
            }
            setTimeout(() => {
                funProgress(id)
            }, 500);
        }, function(error) {
            console.log(error)
            return
        })
    }
    
    funProgress(id)
    res = Array.from(arguments);
    res[0] = id
    res[1] = get_tab_index('mode_img2img')
    return res.slice(0, res.length - 2)
}

function restoreProgressTxt2img() {
    showRestoreProgressButton("img2img", false);
    var id = localGet("img2img_task_id");

    if (id) {
        requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function() {
            showSubmitButtons('img2img', true);
        }, null, 0);
    }

    return id;
}

function restoreProgressImg2img() {
    showRestoreProgressButton("img2img", false);

    var id = localGet("img2img_task_id");

    if (id) {
        requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function() {
            showSubmitButtons('img2img', true);
        }, null, 0);
    }

    return id;
}


onUiLoaded(function() {
    showRestoreProgressButton('txt2img', localGet("txt2img_task_id"));
    showRestoreProgressButton('img2img', localGet("img2img_task_id"));
});


function modelmerger() {
    var id = randomId();
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function() {});

    var res = create_submit_args(arguments);
    res[0] = id;
    return res;
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    var name_ = prompt('Style name:');
    return [name_, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}


var opts = {};
onAfterUiUpdate(function() {
    if (Object.keys(opts).length != 0) return;

    var json_elem = gradioApp().getElementById('settings_json');
    if (json_elem == null) return;

    var textarea = json_elem.querySelector('textarea');
    var jsdata = textarea.value;
    opts = JSON.parse(jsdata);

    executeCallbacks(optionsChangedCallbacks); /*global optionsChangedCallbacks*/

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display = "none";

    setupTokenCounters();

    var show_all_pages = gradioApp().getElementById('settings_show_all_pages');
    var settings_tabs = gradioApp().querySelector('#settings div');
    if (show_all_pages && settings_tabs) {
        settings_tabs.appendChild(show_all_pages);
        show_all_pages.onclick = function() {
            gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
                if (elem.id == "settings_tab_licenses") {
                    return;
                }

                elem.style.display = "block";
            });
        };
    }
});

onOptionsChanged(function() {
    var elem = gradioApp().getElementById('sd_checkpoint_hash');
    var sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    var shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://google.com/search?q=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea, img2img_textarea = undefined;

function restart_reload() {
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';

    var requestPing = function() {
        requestGet("./internal/ping", {}, function(data) {
            location.reload();
        }, function() {
            setTimeout(requestPing, 500);
        });
    };

    setTimeout(requestPing, 2000);

    return [];
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target) {
    let e = new Event("input", {bubbles: true});
    Object.defineProperty(e, "target", {value: target});
    target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click();
}

function currentImg2imgSourceResolution(w, h, scaleBy) {
    var img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
    return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function() {
        gradioApp().getElementById('img2img_update_resize_to').click();
    }, 500);

    return [];

}



function setRandomSeed(elem_id) {
    var input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    var width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    var height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    if (!width || !height) return [];

    var tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}

const inpaintAnything_waitForElement = async (parent, selector, exist) => {
    return new Promise((resolve) => {
        const observer = new MutationObserver(() => {
            if (!!parent.querySelector(selector) != exist) {
                return;
            }
            observer.disconnect();
            resolve(undefined);
        });

        observer.observe(parent, {
            childList: true,
            subtree: true,
        });

        if (!!parent.querySelector(selector) == exist) {
            resolve(undefined);
        }
    });
};

const inpaintAnything_waitForStyle = async (parent, selector, style) => {
    return new Promise((resolve) => {
        const observer = new MutationObserver(() => {
            if (!parent.querySelector(selector) || !parent.querySelector(selector).style[style]) {
                return;
            }
            observer.disconnect();
            resolve(undefined);
        });

        observer.observe(parent, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ["style"],
        });

        if (!!parent.querySelector(selector) && !!parent.querySelector(selector).style[style]) {
            resolve(undefined);
        }
    });
};

const inpaintAnything_timeout = (ms) => {
    return new Promise(function (resolve, reject) {
        setTimeout(() => reject("Timeout"), ms);
    });
};

async function inpaintAnything_sendToInpaint() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(10000)]);

    const waitForElementToBeRemoved = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, false), inpaintAnything_timeout(10000)]);

    const updateGradioImage = async (element, url, name) => {
        const blob = await (await fetch(url)).blob();
        const file = new File([blob], name, { type: "image/png" });
        const dt = new DataTransfer();
        dt.items.add(file);

        function getClearButton() {
            let clearButton = null;
            let clearLabel = null;

            let allButtons = element.querySelectorAll("button");
            if (allButtons.length > 0) {
                for (let button of allButtons) {
                    let label = button.getAttribute("aria-label");
                    if (label && !label.includes("Edit") && !label.includes("Éditer")) {
                        clearButton = button;
                        clearLabel = label;
                        break;
                    }
                }
            }
            return [clearButton, clearLabel];
        }

        const [clearButton, clearLabel] = getClearButton();

        if (clearButton) {
            clearButton?.click();
            await waitForElementToBeRemoved(element, `button[aria-label='${clearLabel}']`);
        }

        const input = element.querySelector("input[type='file']");
        input.value = "";
        input.files = dt.files;
        input.dispatchEvent(
            new Event("change", {
                bubbles: true,
                composed: true,
            })
        );
        await waitForElementToBeInDocument(element, "button");
    };

    const inputImg = document.querySelector("#ia_input_image img");
    const maskImg = document.querySelector("#mask_out_image img");

    if (!inputImg || !maskImg) {
        return;
    }

    const inputImgDataUrl = inputImg.src;
    const maskImgDataUrl = maskImg.src;

    window.scrollTo(0, 0);
    switch_to_img2img_tab(4);

    await waitForElementToBeInDocument(document.querySelector("#img2img_inpaint_upload_tab"), "#img_inpaint_base");

    await updateGradioImage(document.querySelector("#img_inpaint_base"), inputImgDataUrl, "input.png");
    await updateGradioImage(document.querySelector("#img_inpaint_mask"), maskImgDataUrl, "mask.png");
}

async function inpaintAnything_clearSamMask() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(1000)]);

    const elemId = "#ia_sam_image";

    const targetElement = document.querySelector(elemId);
    if (!targetElement) {
        return;
    }
    await waitForElementToBeInDocument(targetElement, "button[aria-label='Clear']");

    targetElement.style.transform = null;
    targetElement.style.zIndex = null;
    targetElement.style.overflow = "auto";

    const samMaskClear = targetElement.querySelector("button[aria-label='Clear']");
    if (!samMaskClear) {
        return;
    }
    const removeImageButton = targetElement.querySelector("button[aria-label='Remove Image']");
    if (!removeImageButton) {
        return;
    }
    samMaskClear?.click();

    if (typeof inpaintAnything_clearSamMask.clickRemoveImage === "undefined") {
        inpaintAnything_clearSamMask.clickRemoveImage = () => {
            targetElement.style.transform = null;
            targetElement.style.zIndex = null;
        };
    } else {
        removeImageButton.removeEventListener("click", inpaintAnything_clearSamMask.clickRemoveImage);
    }
    removeImageButton.addEventListener("click", inpaintAnything_clearSamMask.clickRemoveImage);
}

async function inpaintAnything_clearSelMask() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([inpaintAnything_waitForElement(parent, selector, true), inpaintAnything_timeout(1000)]);

    const elemId = "#ia_sel_mask";

    const targetElement = document.querySelector(elemId);
    if (!targetElement) {
        return;
    }
    await waitForElementToBeInDocument(targetElement, "button[aria-label='Clear']");

    targetElement.style.transform = null;
    targetElement.style.zIndex = null;
    targetElement.style.overflow = "auto";

    const selMaskClear = targetElement.querySelector("button[aria-label='Clear']");
    if (!selMaskClear) {
        return;
    }
    const removeImageButton = targetElement.querySelector("button[aria-label='Remove Image']");
    if (!removeImageButton) {
        return;
    }
    selMaskClear?.click();

    if (typeof inpaintAnything_clearSelMask.clickRemoveImage === "undefined") {
        inpaintAnything_clearSelMask.clickRemoveImage = () => {
            targetElement.style.transform = null;
            targetElement.style.zIndex = null;
        };
    } else {
        removeImageButton.removeEventListener("click", inpaintAnything_clearSelMask.clickRemoveImage);
    }
    removeImageButton.addEventListener("click", inpaintAnything_clearSelMask.clickRemoveImage);
}

async function inpaintAnything_initSamSelMask() {
    inpaintAnything_clearSamMask();
    inpaintAnything_clearSelMask();
}

async function inpaintAnything_getPrompt(tabName, promptId, negPromptId) {
    const tabTxt2img = document.querySelector(`#tab_${tabName}`);
    if (!tabTxt2img) {
        return;
    }

    const txt2imgPrompt = tabTxt2img.querySelector(`#${tabName}_prompt textarea`);
    const txt2imgNegPrompt = tabTxt2img.querySelector(`#${tabName}_neg_prompt textarea`);
    if (!txt2imgPrompt || !txt2imgNegPrompt) {
        return;
    }

    const iaSdPrompt = document.querySelector(`#${promptId} textarea`);
    const iaSdNPrompt = document.querySelector(`#${negPromptId} textarea`);
    if (!iaSdPrompt || !iaSdNPrompt) {
        return;
    }

    iaSdPrompt.value = txt2imgPrompt.value;
    iaSdNPrompt.value = txt2imgNegPrompt.value;

    iaSdPrompt.dispatchEvent(new Event("input", { bubbles: true }));
    iaSdNPrompt.dispatchEvent(new Event("input", { bubbles: true }));
}

async function inpaintAnything_getTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_sd_prompt", "ia_sd_n_prompt");
}

async function inpaintAnything_getImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_sd_prompt", "ia_sd_n_prompt");
}

async function inpaintAnything_webuiGetTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_webui_sd_prompt", "ia_webui_sd_n_prompt");
}

async function inpaintAnything_webuiGetImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_webui_sd_prompt", "ia_webui_sd_n_prompt");
}

async function inpaintAnything_cnGetTxt2imgPrompt() {
    inpaintAnything_getPrompt("txt2img", "ia_cn_sd_prompt", "ia_cn_sd_n_prompt");
}

async function inpaintAnything_cnGetImg2imgPrompt() {
    inpaintAnything_getPrompt("img2img", "ia_cn_sd_prompt", "ia_cn_sd_n_prompt");
}

onUiLoaded(async () => {
    const elementIDs = {
        ia_sam_image: "#ia_sam_image",
        ia_sel_mask: "#ia_sel_mask",
        ia_out_image: "#ia_out_image",
        ia_cleaner_out_image: "#ia_cleaner_out_image",
        ia_webui_out_image: "#ia_webui_out_image",
        ia_cn_out_image: "#ia_cn_out_image",
    };

    function setStyleHeight(elemId, height) {
        const elem = gradioApp().querySelector(elemId);
        if (elem) {
            if (!elem.style.height) {
                elem.style.height = height;
                const observer = new MutationObserver(() => {
                    const divPreview = elem.querySelector(".preview");
                    if (divPreview) {
                        divPreview.classList.remove("fixed-height");
                    }
                });
                observer.observe(elem, {
                    childList: true,
                    attributes: true,
                    attributeFilter: ["class"],
                });
            }
        }
    }

    setStyleHeight(elementIDs.ia_out_image, "520px");
    setStyleHeight(elementIDs.ia_cleaner_out_image, "520px");
    setStyleHeight(elementIDs.ia_webui_out_image, "520px");
    setStyleHeight(elementIDs.ia_cn_out_image, "520px");

    // Default config
    const defaultHotkeysConfig = {
        canvas_hotkey_reset: "KeyR",
        canvas_hotkey_fullscreen: "KeyS",
    };

    const elemData = {};
    let activeElement;

    function applyZoomAndPan(elemId) {
        const targetElement = gradioApp().querySelector(elemId);

        if (!targetElement) {
            console.log("Element not found");
            return;
        }

        targetElement.style.transformOrigin = "0 0";

        elemData[elemId] = {
            zoomLevel: 1,
            panX: 0,
            panY: 0,
        };
        let fullScreenMode = false;

        // Toggle the zIndex of the target element between two values, allowing it to overlap or be overlapped by other elements
        function toggleOverlap(forced = "") {
            // const zIndex1 = "0";
            const zIndex1 = null;
            const zIndex2 = "998";

            targetElement.style.zIndex = targetElement.style.zIndex !== zIndex2 ? zIndex2 : zIndex1;

            if (forced === "off") {
                targetElement.style.zIndex = zIndex1;
            } else if (forced === "on") {
                targetElement.style.zIndex = zIndex2;
            }
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        function fitToElement() {
            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const parentElement = targetElement.parentElement;
            const screenWidth = parentElement.clientWidth;
            const screenHeight = parentElement.clientHeight;

            // Get element's coordinates relative to the parent element
            const elementRect = targetElement.getBoundingClientRect();
            const parentRect = parentElement.getBoundingClientRect();
            const elementX = elementRect.x - parentRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            const transformOrigin = window.getComputedStyle(targetElement).transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            const offsetX = (screenWidth - elementWidth * scale) / 2 - originXValue * (1 - scale);
            const offsetY = (screenHeight - elementHeight * scale) / 2.5 - originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = false;
            toggleOverlap("off");
        }

        // Reset the zoom level and pan position of the target element to their initial values
        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0,
            };

            // fixCanvas();
            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;

            // const canvas = gradioApp().querySelector(`${elemId} canvas[key="interface"]`);

            toggleOverlap("off");
            fullScreenMode = false;

            // if (
            //     canvas &&
            //     parseFloat(canvas.style.width) > 865 &&
            //     parseFloat(targetElement.style.width) > 865
            // ) {
            //     fitToElement();
            //     return;
            // }

            // targetElement.style.width = "";
            // if (canvas) {
            //     targetElement.style.height = canvas.style.height;
            // }
            targetElement.style.width = null;
            targetElement.style.height = 480;
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        // Fullscreen mode
        function fitToScreen() {
            const canvas = gradioApp().querySelector(`${elemId} canvas[key="interface"]`);
            const img = gradioApp().querySelector(`${elemId} img`);

            if (!canvas && !img) return;

            // if (canvas.offsetWidth > 862) {
            //     targetElement.style.width = canvas.offsetWidth + "px";
            // }

            if (fullScreenMode) {
                resetZoom();
                fullScreenMode = false;
                return;
            }

            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get scrollbar width to right-align the image
            const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const screenWidth = window.innerWidth - scrollbarWidth;
            const screenHeight = window.innerHeight;

            // Get element's coordinates relative to the page
            const elementRect = targetElement.getBoundingClientRect();
            const elementY = elementRect.y;
            const elementX = elementRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            // Get the current transformOrigin
            const computedStyle = window.getComputedStyle(targetElement);
            const transformOrigin = computedStyle.transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            // Calculate offsets with respect to the transformOrigin
            const offsetX = (screenWidth - elementWidth * scale) / 2 - elementX - originXValue * (1 - scale);
            const offsetY = (screenHeight - elementHeight * scale) / 2 - elementY - originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = true;
            toggleOverlap("on");
        }

        // Reset zoom when uploading a new image
        const fileInput = gradioApp().querySelector(`${elemId} input[type="file"][accept="image/*"].svelte-116rqfv`);
        if (fileInput) {
            fileInput.addEventListener("click", resetZoom);
        }

        // Handle keydown events
        function handleKeyDown(event) {
            // Disable key locks to make pasting from the buffer work correctly
            if (
                (event.ctrlKey && event.code === "KeyV") ||
                (event.ctrlKey && event.code === "KeyC") ||
                event.code === "F5"
            ) {
                return;
            }

            // before activating shortcut, ensure user is not actively typing in an input field
            if (event.target.nodeName === "TEXTAREA" || event.target.nodeName === "INPUT") {
                return;
            }

            const hotkeyActions = {
                [defaultHotkeysConfig.canvas_hotkey_reset]: resetZoom,
                [defaultHotkeysConfig.canvas_hotkey_fullscreen]: fitToScreen,
            };

            const action = hotkeyActions[event.code];
            if (action) {
                event.preventDefault();
                action(event);
            }
        }

        // Handle events only inside the targetElement
        let isKeyDownHandlerAttached = false;

        function handleMouseMove() {
            if (!isKeyDownHandlerAttached) {
                document.addEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = true;

                activeElement = elemId;
            }
        }

        function handleMouseLeave() {
            if (isKeyDownHandlerAttached) {
                document.removeEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = false;

                activeElement = null;
            }
        }

        // Add mouse event handlers
        targetElement.addEventListener("mousemove", handleMouseMove);
        targetElement.addEventListener("mouseleave", handleMouseLeave);
    }

    applyZoomAndPan(elementIDs.ia_sam_image);
    applyZoomAndPan(elementIDs.ia_sel_mask);
    // applyZoomAndPan(elementIDs.ia_out_image);
    // applyZoomAndPan(elementIDs.ia_cleaner_out_image);
    // applyZoomAndPan(elementIDs.ia_webui_out_image);
    // applyZoomAndPan(elementIDs.ia_cn_out_image);
});
