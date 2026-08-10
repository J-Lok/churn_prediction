document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // 1. Tab Switching Logic
    // ----------------------------------------------------
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTabId = btn.getAttribute('data-tab');
            
            // Toggle active button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle active content
            tabContents.forEach(content => {
                if (content.id === targetTabId) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
            
            // If switching to performance or insights, we might need to redraw charts
            if (targetTabId === 'tab-analytics' || targetTabId === 'tab-insights') {
                loadPerformanceMetrics();
            }
        });
    });

    // ----------------------------------------------------
    // 2. Synchronize Form Sliders & Auto-Estimates
    // ----------------------------------------------------
    const tenureSlider = document.getElementById('tenure');
    const tenureVal = document.getElementById('tenure-val');
    const monthlySlider = document.getElementById('MonthlyCharges');
    const monthlyVal = document.getElementById('MonthlyCharges-val');
    const totalInput = document.getElementById('TotalCharges');
    const calcTotalBtn = document.getElementById('calc-total-btn');

    // Update labels on slider move
    tenureSlider.addEventListener('input', (e) => {
        tenureVal.textContent = e.target.value;
        autoEstimateTotal();
    });

    monthlySlider.addEventListener('input', (e) => {
        monthlyVal.textContent = parseFloat(e.target.value).toFixed(2);
        autoEstimateTotal();
    });

    // Auto estimate total charges: tenure * monthly charges
    function autoEstimateTotal() {
        const tenure = parseInt(tenureSlider.value);
        const monthly = parseFloat(monthlySlider.value);
        const total = (tenure * monthly).toFixed(2);
        totalInput.value = Math.round(total);
        document.getElementById('TotalCharges-val').textContent = parseFloat(total).toFixed(2);
    }

    // Force recalculate button
    calcTotalBtn.addEventListener('click', autoEstimateTotal);

    // Initial run
    autoEstimateTotal();

    // ----------------------------------------------------
    // 3. Form Submission & Real-time Prediction API
    // ----------------------------------------------------
    const form = document.getElementById('predictor-form');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultDisplay = document.getElementById('result-display');
    const submitBtn = form.querySelector('.submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state on button
        const originalBtnHTML = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyse en cours...';
        submitBtn.disabled = true;

        // Gather form data
        const formData = new FormData(form);
        const payload = {
            gender: formData.get('gender'),
            SeniorCitizen: parseInt(formData.get('SeniorCitizen')),
            Partner: formData.get('Partner'),
            Dependents: formData.get('Dependents'),
            tenure: parseInt(formData.get('tenure')),
            PhoneService: formData.get('PhoneService'),
            MultipleLines: formData.get('MultipleLines'),
            InternetService: formData.get('InternetService'),
            OnlineSecurity: formData.get('OnlineSecurity'),
            OnlineBackup: formData.get('OnlineBackup'),
            DeviceProtection: formData.get('DeviceProtection'),
            TechSupport: formData.get('TechSupport'),
            StreamingTV: formData.get('StreamingTV'),
            StreamingMovies: formData.get('StreamingMovies'),
            Contract: formData.get('Contract'),
            PaperlessBilling: formData.get('PaperlessBilling'),
            PaymentMethod: formData.get('PaymentMethod'),
            MonthlyCharges: parseFloat(formData.get('MonthlyCharges')),
            TotalCharges: formData.get('TotalCharges').toString()
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Erreur réseau lors de la récupération des predictions.');
            }

            const data = await response.json();
            displayPredictionResults(data, payload);
            
        } catch (error) {
            console.error(error);
            alert('Erreur: ' + error.message);
        } finally {
            // Restore button state
            submitBtn.innerHTML = originalBtnHTML;
            submitBtn.disabled = false;
        }
    });

    // Animate Circular Progress Gauge & details
    function displayPredictionResults(data, inputs) {
        // Toggle view containers
        resultPlaceholder.classList.add('hidden');
        resultDisplay.classList.remove('hidden');

        // Scroll to results on mobile devices
        if (window.innerWidth <= 1024) {
            resultDisplay.scrollIntoView({ behavior: 'smooth' });
        }

        // Primary Model (Logistic Regression) Risk Gauge
        const lrProb = data.logistic_regression.probability;
        const lrPercentStr = (lrProb * 100).toFixed(1) + '%';
        document.getElementById('lr-prob-val').textContent = lrPercentStr;

        // Circular dash offset animation
        const circle = document.getElementById('lr-fill-circle');
        const circumference = 2 * Math.PI * 42; // r=42 -> ~263.89
        const offset = circumference - (lrProb * circumference);
        circle.style.strokeDashoffset = offset;

        // Dynamic coloring of circular gauge and risk badge
        const riskLabel = document.getElementById('lr-risk-label');
        riskLabel.className = 'risk-label'; // Reset classes
        
        if (lrProb < 0.3) {
            circle.style.stroke = '#10b981'; // Emerald green
            riskLabel.textContent = 'RISQUE FAIBLE';
            riskLabel.classList.add('risk-low');
        } else if (lrProb < 0.6) {
            circle.style.stroke = '#f59e0b'; // Amber Orange
            riskLabel.textContent = 'RISQUE MODÉRÉ';
            riskLabel.classList.add('risk-medium');
        } else {
            circle.style.stroke = '#ef4444'; // Red
            riskLabel.textContent = 'RISQUE ÉLEVÉ';
            riskLabel.classList.add('risk-high');
        }

        // Other models update
        const nnProb = data.neural_network.probability;
        document.getElementById('nn-prob-val').textContent = (nnProb * 100).toFixed(1) + '%';
        document.getElementById('nn-progress-fill').style.width = (nnProb * 100) + '%';

        const knnProb = data.knn.probability;
        document.getElementById('knn-prob-val').textContent = (knnProb * 100).toFixed(1) + '%';
        document.getElementById('knn-progress-fill').style.width = (knnProb * 100) + '%';

        // Local factor analysis based on customer variables
        analyzeIndividualFactors(inputs);
    }

    function analyzeIndividualFactors(inputs) {
        const retentionList = document.getElementById('retention-list');
        const riskList = document.getElementById('risk-list');

        retentionList.innerHTML = '';
        riskList.innerHTML = '';

        const retentionFactors = [];
        const riskFactors = [];

        // Contract type
        if (inputs.Contract === 'Month-to-month') {
            riskFactors.push({ icon: 'fa-file-invoice', text: 'Contrat sans engagement (mensuel)' });
        } else if (inputs.Contract === 'Two year') {
            retentionFactors.push({ icon: 'fa-shield-halved', text: 'Engagement long terme (2 ans)' });
        } else {
            retentionFactors.push({ icon: 'fa-shield-halved', text: 'Engagement moyen terme (1 an)' });
        }

        // Tenure
        if (inputs.tenure >= 40) {
            retentionFactors.push({ icon: 'fa-calendar-check', text: 'Grande ancienneté (> 3 ans)' });
        } else if (inputs.tenure <= 6) {
            riskFactors.push({ icon: 'fa-hourglass-start', text: 'Nouveau client (ancienneté ≤ 6 mois)' });
        }

        // Internet type
        if (inputs.InternetService === 'Fiber optic') {
            riskFactors.push({ icon: 'fa-network-wired', text: 'Abonné Fibre Optique (Frais/Départ élevé)' });
        } else if (inputs.InternetService === 'No') {
            retentionFactors.push({ icon: 'fa-ban', text: 'Aucun abonnement Internet' });
        }

        // Security / Assistance services
        if (inputs.InternetService !== 'No') {
            if (inputs.OnlineSecurity === 'Yes') {
                retentionFactors.push({ icon: 'fa-user-shield', text: 'Sécurité en ligne activée' });
            } else {
                riskFactors.push({ icon: 'fa-shield', text: 'Pas de protection de sécurité en ligne' });
            }

            if (inputs.TechSupport === 'Yes') {
                retentionFactors.push({ icon: 'fa-circle-question', text: 'Bénéficie du support technique' });
            } else {
                riskFactors.push({ icon: 'fa-headset', text: 'Pas d\'assistance technique' });
            }
        }

        // Payment method
        if (inputs.PaymentMethod === 'Electronic check') {
            riskFactors.push({ icon: 'fa-money-bill-transfer', text: 'Paiement par chèque électronique' });
        } else if (inputs.PaymentMethod === 'Credit card' || inputs.PaymentMethod === 'Bank transfer') {
            retentionFactors.push({ icon: 'fa-credit-card', text: 'Prélèvement automatique configuré' });
        }

        // Render retention list
        if (retentionFactors.length > 0) {
            retentionFactors.forEach(f => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fa-solid ${f.icon} text-green"></i> ${f.text}`;
                retentionList.appendChild(li);
            });
        } else {
            retentionList.innerHTML = '<li>Aucun facteur protecteur majeur détecté.</li>';
        }

        // Render risk list
        if (riskFactors.length > 0) {
            riskFactors.forEach(f => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fa-solid ${f.icon} text-red"></i> ${f.text}`;
                riskList.appendChild(li);
            });
        } else {
            riskList.innerHTML = '<li>Aucun facteur de risque majeur détecté.</li>';
        }
    }

    // ----------------------------------------------------
    // 4. Load Performance Metrics & Curves from API
    // ----------------------------------------------------
    let metricsLoaded = false;
    let rocChart = null;
    let prChart = null;
    let importanceChart = null;

    async function loadPerformanceMetrics() {
        if (metricsLoaded) return; // Load only once

        try {
            const response = await fetch('/api/metrics');
            if (!response.ok) {
                throw new Error('Erreur lors du chargement des métriques globales.');
            }

            const data = await response.json();
            
            // Populate metrics comparison table
            populateMetricsTable(data);

            // Populate confusion matrices
            renderConfusionMatrices(data);

            // Render Charts
            renderRocAndPrCharts(data);
            renderFeatureImportanceChart(data.feature_importances);

            metricsLoaded = true;
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }

    function populateMetricsTable(data) {
        const tbody = document.getElementById('metrics-table-body');
        tbody.innerHTML = '';

        const models = [
            { id: 'logistic_regression', name: 'Régression Logistique (Principal)' },
            { id: 'neural_network', name: 'Réseau de Neurones (MLP)' },
            { id: 'knn', name: 'K-Nearest Neighbors' }
        ];

        models.forEach(m => {
            const mData = data[m.id];
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="model-name-col">${m.name}</td>
                <td>${(mData.accuracy * 100).toFixed(2)}%</td>
                <td>${(mData.f1_score * 100).toFixed(2)}%</td>
                <td>${mData.roc_auc.toFixed(4)}</td>
                <td>${mData.pr_auc.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderConfusionMatrices(data) {
        const container = document.getElementById('cm-container');
        container.innerHTML = '';

        const models = [
            { id: 'logistic_regression', name: 'Régression Logistique' },
            { id: 'neural_network', name: 'Réseau de Neurones' },
            { id: 'knn', name: 'K-Nearest Neighbors' }
        ];

        models.forEach(m => {
            const cm = data[m.id].confusion_matrix;
            // cm standard structure: [[TN, FP], [FN, TP]]
            const tn = cm[0][0];
            const fp = cm[0][1];
            const fn = cm[1][0];
            const tp = cm[1][1];
            const total = tn + fp + fn + tp;

            const box = document.createElement('div');
            box.className = 'cm-box';
            box.innerHTML = `
                <h3>${m.name}</h3>
                <div class="cm-grid">
                    <!-- Row Headers -->
                    <div class="cm-label">Actuel</div>
                    <div class="cm-label">Prédit: NON</div>
                    <div class="cm-label">Prédit: OUI</div>
                    
                    <!-- Row 1: No Churn -->
                    <div class="cm-label">Fidèle</div>
                    <div class="cm-cell cm-true-neg">
                        <span class="cm-val">${tn}</span>
                        <span class="cm-lbl">VN (${((tn/total)*100).toFixed(1)}%)</span>
                    </div>
                    <div class="cm-cell cm-false-pos">
                        <span class="cm-val">${fp}</span>
                        <span class="cm-lbl">FP (${((fp/total)*100).toFixed(1)}%)</span>
                    </div>
                    
                    <!-- Row 2: Churn -->
                    <div class="cm-label">Désabonné</div>
                    <div class="cm-cell cm-false-neg">
                        <span class="cm-val">${fn}</span>
                        <span class="cm-lbl">FN (${((fn/total)*100).toFixed(1)}%)</span>
                    </div>
                    <div class="cm-cell cm-true-pos">
                        <span class="cm-val">${tp}</span>
                        <span class="cm-lbl">VP (${((tp/total)*100).toFixed(1)}%)</span>
                    </div>
                </div>
            `;
            container.appendChild(box);
        });
    }

    function renderRocAndPrCharts(data) {
        // Colors mapping
        const colors = {
            logistic_regression: { stroke: '#06b6d4', bg: 'rgba(6, 182, 212, 0.1)' },
            neural_network: { stroke: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.1)' },
            knn: { stroke: '#e11d48', bg: 'rgba(225, 29, 72, 0.1)' }
        };

        // 1. ROC Chart
        const rocCtx = document.getElementById('roc-chart').getContext('2d');
        
        const rocDatasets = [
            {
                label: 'Modèle Aléatoire (AUC = 0.5)',
                data: [{x: 0, y: 0}, {x: 1, y: 1}],
                borderColor: 'rgba(148, 163, 184, 0.4)',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }
        ];

        const models = [
            { id: 'logistic_regression', name: 'Régression Logistique' },
            { id: 'neural_network', name: 'Réseau de Neurones' },
            { id: 'knn', name: 'K-Nearest Neighbors' }
        ];

        models.forEach(m => {
            const curve = data[m.id].roc_curve;
            const points = curve.fpr.map((fprVal, idx) => {
                return { x: fprVal, y: curve.tpr[idx] };
            });

            rocDatasets.push({
                label: `${m.name} (AUC = ${data[m.id].roc_auc.toFixed(3)})`,
                data: points,
                borderColor: colors[m.id].stroke,
                backgroundColor: colors[m.id].bg,
                borderWidth: 2.5,
                fill: false,
                tension: 0.2,
                pointRadius: 1
            });
        });

        rocChart = new Chart(rocCtx, {
            type: 'line',
            data: { datasets: rocDatasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Taux de Faux Positifs (FPR)', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        type: 'linear',
                        title: { display: true, text: 'Taux de Vrais Positifs (TPR)', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc', font: { family: 'Inter' } } }
                }
            }
        });

        // 2. Precision-Recall Chart
        const prCtx = document.getElementById('pr-chart').getContext('2d');
        const prDatasets = [];

        models.forEach(m => {
            const curve = data[m.id].pr_curve;
            const points = curve.recall.map((recVal, idx) => {
                return { x: recVal, y: curve.precision[idx] };
            });

            prDatasets.push({
                label: `${m.name} (PR AUC = ${data[m.id].pr_auc.toFixed(3)})`,
                data: points,
                borderColor: colors[m.id].stroke,
                backgroundColor: colors[m.id].bg,
                borderWidth: 2.5,
                fill: false,
                tension: 0.2,
                pointRadius: 1
            });
        });

        prChart = new Chart(prCtx, {
            type: 'line',
            data: { datasets: prDatasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Rappel (Recall)', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        type: 'linear',
                        title: { display: true, text: 'Précision (Precision)', color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc', font: { family: 'Inter' } } }
                }
            }
        });
    }

    function renderFeatureImportanceChart(importances) {
        const ctx = document.getElementById('importance-chart').getContext('2d');
        
        // Sort importances by absolute value of weights (we already did this in backend, but keep first 12 for chart)
        const activeImportances = importances.slice(0, 12);
        
        const labels = activeImportances.map(item => {
            // Translate feature names to French descriptions for high-end feel
            let feat = item.feature;
            feat = feat.replace('Contract_Month-to-month', 'Contrat Mensuel');
            feat = feat.replace('Contract_Two year', 'Contrat 2 Ans');
            feat = feat.replace('Contract_One year', 'Contrat 1 An');
            feat = feat.replace('InternetService_Fiber optic', 'Internet Fibre Optique');
            feat = feat.replace('InternetService_DSL', 'Internet DSL');
            feat = feat.replace('InternetService_No', 'Pas d\'Internet');
            feat = feat.replace('PaymentMethod_Electronic check', 'Chèque électronique');
            feat = feat.replace('PaymentMethod_Credit card', 'Carte de crédit');
            feat = feat.replace('PaymentMethod_Mailed check', 'Chèque postal');
            feat = feat.replace('PaymentMethod_Bank transfer', 'Virement bancaire');
            feat = feat.replace('OnlineSecurity_No internet service', 'Pas d\'Internet (Sécurité)');
            feat = feat.replace('OnlineSecurity_No', 'Pas de sécurité en ligne');
            feat = feat.replace('OnlineSecurity_Yes', 'Sécurité en ligne active');
            feat = feat.replace('TechSupport_No internet service', 'Pas d\'Internet (Support)');
            feat = feat.replace('TechSupport_No', 'Pas de support technique');
            feat = feat.replace('TechSupport_Yes', 'Support technique actif');
            feat = feat.replace('PaperlessBilling', 'Facturation sans papier');
            feat = feat.replace('SeniorCitizen', 'Client Senior');
            feat = feat.replace('gender', 'Genre (Femme=1)');
            feat = feat.replace('tenure', 'Ancienneté');
            feat = feat.replace('MonthlyCharges', 'Frais Mensuels');
            feat = feat.replace('TotalCharges', 'Frais Totaux');
            return feat;
        });
        
        const values = activeImportances.map(item => item.weight);
        
        // Dynamic colors: red for positive weights (risk), blue for negative weights (retention)
        const backgroundColors = values.map(val => val > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(6, 182, 212, 0.7)');
        const borderColors = values.map(val => val > 0 ? '#ef4444' : '#06b6d4');

        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Poids du coefficient (Régression Logistique)',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' },
                        title: { display: true, text: 'Poids du coefficient', color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f8fafc', font: { family: 'Outfit', weight: 500 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // ----------------------------------------------------
    // 5. Automated Demo/Routing Mode for Screen Capture
    // ----------------------------------------------------
    const urlParams = new URLSearchParams(window.location.search);
    const demoParam = urlParams.get('demo');
    const tabParam = urlParams.get('tab');

    if (tabParam) {
        const btn = document.querySelector(`.tab-btn[data-tab="tab-${tabParam}"]`);
        if (btn) {
            setTimeout(() => {
                btn.click();
            }, 300);
        }
    }

    if (demoParam) {
        // Populate form fields with a high-risk sample
        setTimeout(() => {
            document.getElementById('gender').value = 'Female';
            document.getElementById('SeniorCitizen').value = '1';
            document.getElementById('Partner').value = 'No';
            document.getElementById('Dependents').value = 'No';
            document.getElementById('PhoneService').value = 'Yes';
            document.getElementById('MultipleLines').value = 'Yes';
            document.getElementById('InternetService').value = 'Fiber optic';
            document.getElementById('OnlineSecurity').value = 'No';
            document.getElementById('OnlineBackup').value = 'No';
            document.getElementById('DeviceProtection').value = 'No';
            document.getElementById('TechSupport').value = 'No';
            document.getElementById('StreamingTV').value = 'Yes';
            document.getElementById('StreamingMovies').value = 'Yes';
            document.getElementById('Contract').value = 'Month-to-month';
            document.getElementById('PaperlessBilling').value = 'Yes';
            document.getElementById('PaymentMethod').value = 'Electronic check';
            document.getElementById('tenure').value = '2';
            document.getElementById('tenure-val').textContent = '2';
            document.getElementById('MonthlyCharges').value = '98.50';
            document.getElementById('MonthlyCharges-val').textContent = '98.50';
            document.getElementById('TotalCharges').value = '197';
            
            // Trigger submit
            setTimeout(() => {
                form.dispatchEvent(new Event('submit'));
            }, 200);
        }, 500);
    }
});
