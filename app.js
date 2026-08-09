document.addEventListener("DOMContentLoaded", function () {

    const fileInput =
        document.getElementById("fileInput");

    const imagePreview =
        document.getElementById("imagePreview");

    const previewContainer =
        document.getElementById("previewContainer");

    const uploadForm =
        document.getElementById("uploadForm");

    const loading =
        document.getElementById("loading");


    // =====================================================
    // IMAGE PREVIEW
    // =====================================================

    if (fileInput) {

        fileInput.addEventListener(
            "change",
            function () {

                const file = this.files[0];

                if (!file) {

                    if (imagePreview) {
                        imagePreview.style.display = "none";
                    }

                    if (previewContainer) {
                        previewContainer.style.display = "none";
                    }

                    return;
                }


                // Make sure the selected file is an image

                if (!file.type.startsWith("image/")) {

                    alert(
                        "Please select a valid image file."
                    );

                    this.value = "";

                    return;
                }


                const reader =
                    new FileReader();


                reader.onload = function (event) {

                    if (imagePreview) {

                        imagePreview.src =
                            event.target.result;

                        imagePreview.style.display =
                            "block";
                    }


                    if (previewContainer) {

                        previewContainer.style.display =
                            "block";
                    }

                };


                reader.readAsDataURL(file);

            }
        );

    }


    // =====================================================
    // FORM SUBMISSION
    // =====================================================

    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            function () {

                if (loading) {

                    loading.style.display =
                        "block";

                }

            }
        );

    }

});