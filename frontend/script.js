const API_URL = 'http://localhost:8000';

const inputDisplay = document.getElementById('input-display');
const outputDisplay = document.getElementById('output-display');
const restoreBtn = document.getElementById('restore-btn');
const galleryGrid = document.getElementById('gallery-grid');
const loader = document.getElementById('loader');

let selectedImageName = null;

// Load Gallery
async function loadGallery() {
    try {
        const response = await fetch(`${API_URL}/gallery`);
        const data = await response.json();

        galleryGrid.innerHTML = '';
        data.images.forEach(imgName => {
            const div = document.createElement('div');
            div.className = 'gallery-item';
            div.innerHTML = `<img src="${API_URL}/static/demo_images/${imgName}" alt="Demo">`;
            div.onclick = () => selectGalleryImage(imgName, div);
            galleryGrid.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading gallery:', error);
    }
}

function selectGalleryImage(imgName, element) {
    document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    selectedImageName = imgName;
    inputDisplay.src = `${API_URL}/static/demo_images/${imgName}`;
    outputDisplay.src = `https://via.placeholder.com/512?text=Ready+to+Restore`;
}

// Handle Restoration
restoreBtn.onclick = async () => {
    if (!selectedImageName) {
        alert('Please select a distorted image from the gallery first!');
        return;
    }

    loader.style.display = 'block';
    restoreBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image_name', selectedImageName);

        const response = await fetch(`${API_URL}/restore`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.restored_image) {
            outputDisplay.src = data.restored_image;
        } else {
            alert('Restoration failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the backend server.');
    } finally {
        loader.style.display = 'none';
        restoreBtn.disabled = false;
    }
};

// Initialize
loadGallery();
