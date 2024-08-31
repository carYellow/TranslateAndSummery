document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('summarize-form');
    const summaryDiv = document.getElementById('summary');
    const loadingDiv = document.getElementById('loading');

    // Update range input values
    ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty'].forEach(param => {
        const input = document.getElementById(param);
        const value = document.getElementById(`${param}-value`);
        input.addEventListener('input', () => {
            value.textContent = input.value;
        });
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const text = document.getElementById('text').value;
        const temperature = document.getElementById('temperature').value;
        const max_tokens = document.getElementById('max_tokens').value;
        const top_p = document.getElementById('top_p').value;
        const frequency_penalty = document.getElementById('frequency_penalty').value;
        const presence_penalty = document.getElementById('presence_penalty').value;

        summaryDiv.innerHTML = '';
        loadingDiv.style.display = 'block';

        const params = new URLSearchParams({
            text,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty
        });

        const response = await fetch(`/summarize?${params}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.he) {
                        const bulletPoint = document.createElement('li');
                        bulletPoint.textContent = data.he;
                        summaryDiv.appendChild(bulletPoint);
                    }
                }
            }
        }

        loadingDiv.style.display = 'none';
    });
});document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('summarize-form');
    const summaryDiv = document.getElementById('summary');
    const loadingDiv = document.getElementById('loading');

    // Update range input values
    ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty'].forEach(param => {
        const input = document.getElementById(param);
        const value = document.getElementById(`${param}-value`);
        input.addEventListener('input', () => {
            value.textContent = input.value;
        });
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const text = document.getElementById('text').value;
        const temperature = document.getElementById('temperature').value;
        const max_tokens = document.getElementById('max_tokens').value;
        const top_p = document.getElementById('top_p').value;
        const frequency_penalty = document.getElementById('frequency_penalty').value;
        const presence_penalty = document.getElementById('presence_penalty').value;

        summaryDiv.innerHTML = '';
        loadingDiv.style.display = 'block';

        const params = new URLSearchParams({
            text,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty
        });

        const response = await fetch(`/summarize?${params}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.he) {
                        const bulletPoint = document.createElement('li');
                        bulletPoint.textContent = data.he;
                        summaryDiv.appendChild(bulletPoint);
                    }
                }
            }
        }

        loadingDiv.style.display = 'none';
    });
});