# WiFi Sensing Paper - Interactive Project Page

This is the GitHub Pages site for the paper:

**"Efficient WiFi Sensing-Based Human Activity Recognition With WiFi-Enabled Access Points"**

## Authors
- Gad Gad, Iqra Batool, Mostafa M. Fouda, Shikhar Verma, Zubair Md Fadlullah

## Conference
IEEE International Conference on Communications (ICC) 2026

## Features

- **Conference-paper inspired layout** with two-column sections
- **Animated tables** with scroll-triggered row animations
- **Interactive bar charts** comparing ML and DL performance
- **Zoomable PCA visualizations** of dataset distributions
- **Animated system architecture diagram** showing the signal model and processing pipeline
- **LaTeX math rendering** using KaTeX for equations
- **Responsive design** for mobile and desktop viewing

## Deployment

### GitHub Pages Setup

1. Go to your repository settings on GitHub
2. Navigate to "Pages" section
3. Under "Source", select "Deploy from a branch"
4. Choose `main` branch and `/docs` folder
5. Save and wait for deployment

Your site will be available at: `https://[username].github.io/newpapercode/`

## Local Development

To preview locally, you can use any static file server:

```bash
# Using Python
cd docs
python -m http.server 8000

# Using Node.js (npx)
npx serve docs

# Using PHP
php -S localhost:8000 -t docs
```

Then open `http://localhost:8000` in your browser.

## File Structure

```
docs/
├── index.html      # Main HTML page
├── styles.css      # All styling and animations
├── script.js       # Interactive JavaScript
├── README.md       # This file
└── assets/         # Images from the paper
    ├── pca_2d_scatter.png
    ├── pca_home_har.png
    ├── pca_home_occupation.png
    ├── pca_office_har.png
    ├── pca_office_localization.png
    ├── pca_iris.png
    └── pca_mnist.png
```

## Key Results Highlighted

- **Rolling Variance**: 1.6× higher accuracy than raw amplitude
- **100× faster** than SHARP-based phase sanitisation
- **1D-CNN** achieves 52.9% on challenging 7-class Home HAR
- **No phase information needed** - works with amplitude-only CSI

## License

This project page is for academic purposes accompanying the IEEE ICC 2026 paper.
