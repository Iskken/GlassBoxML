pipeline {
    agent any

    stages {
        stage('Checkout') {
            // Fetch the exact revision that triggered this Jenkins run
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            // Create an isolated virtualenv in the workspace and install
            // glassboxml (editable) plus pytest into it, mirroring the
            // "Install dependencies" step in .github/workflows/tests.yml
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install --upgrade pip
                    .venv/bin/pip install -e ".[test]"
                '''
            }
        }

        stage('Run Tests') {
            // Same command CI runs: the full pytest suite
            steps {
                sh '.venv/bin/pytest'
            }
        }

        stage('Build Docker Image') {
            // Validate the Dockerfile at the repo root still builds cleanly,
            // same command used locally and in the docker-build CI job
            steps {
                sh 'docker build -t glassboxml-webapp .'
            }
        }
    }
}
